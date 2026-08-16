use std::{
    collections::{HashMap, HashSet},
    time::Duration,
};

use gpui::{
    App, ClipboardItem, Context, Entity, EntityId, EventEmitter, FocusHandle, Focusable, Global,
    Render, SharedString, StatefulInteractiveElement, Subscription, Task, Window, actions,
};
use project::{LspStore, LspStoreEvent, Project};
use sysinfo::{Pid, ProcessRefreshKind, ProcessesToUpdate, System};
use ui::{Tooltip, prelude::*};
use workspace::{Item, ItemHandle, SplitDirection, StatusItemView, Workspace, WorkspaceId};

use crate::get_or_create_tool;

const DETAIL_POLL_INTERVAL: Duration = Duration::from_secs(3);
const FOREGROUND_POLL_INTERVAL: Duration = Duration::from_secs(5);
const BACKGROUND_POLL_INTERVAL: Duration = Duration::from_secs(15);

actions!(
    dev,
    [
        /// Opens a live view of CPU and memory used by Zed and its language servers.
        OpenResourceMonitor
    ]
);

pub fn init(cx: &mut App) {
    let metrics = cx.new(ResourceMetrics::new);
    metrics.update(cx, |metrics, cx| metrics.set_enabled(true, cx));
    cx.set_global(GlobalResourceMetrics(metrics));

    cx.observe_new(|workspace: &mut Workspace, _, _| {
        workspace.register_action(|workspace, _: &OpenResourceMonitor, window, cx| {
            let metrics = ResourceMetrics::global(cx);
            get_or_create_tool(
                workspace,
                SplitDirection::Right,
                window,
                cx,
                move |window, cx| ResourceMonitor::new(metrics, window, cx),
            );
        });
    })
    .detach();
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum ProcessKind {
    Zed,
    LanguageServer,
}

#[derive(Clone)]
struct ProcessTarget {
    name: SharedString,
    pid: u32,
    kind: ProcessKind,
}

#[derive(Clone)]
struct ProcessSnapshot {
    name: SharedString,
    pid: u32,
    cpu_percent: f32,
    memory_bytes: u64,
    kind: ProcessKind,
}

struct ProjectProcesses {
    targets: Vec<ProcessTarget>,
    consumers: usize,
    _lsp_subscription: Subscription,
}

struct ResourceMetrics {
    projects: HashMap<EntityId, ProjectProcesses>,
    snapshots: Vec<ProcessSnapshot>,
    has_cpu_sample: bool,
    enabled: bool,
    detail_consumers: usize,
    poll_task: Option<Task<()>>,
}

impl ResourceMetrics {
    fn global(cx: &App) -> Entity<Self> {
        cx.global::<GlobalResourceMetrics>().0.clone()
    }

    fn new(_cx: &mut Context<Self>) -> Self {
        Self {
            projects: HashMap::new(),
            snapshots: Vec::new(),
            has_cpu_sample: false,
            enabled: false,
            detail_consumers: 0,
            poll_task: None,
        }
    }

    fn register_project(&mut self, project: &Entity<Project>, cx: &mut Context<Self>) {
        let project_id = project.entity_id();
        if let Some(registration) = self.projects.get_mut(&project_id) {
            registration.consumers = registration.consumers.saturating_add(1);
            self.ensure_polling(cx);
            return;
        }

        let lsp_store = project.read(cx).lsp_store();
        let targets = lsp_targets(&lsp_store.read(cx));
        let subscription = cx.subscribe(&lsp_store, move |this, lsp_store, event, cx| {
            if matches!(
                event,
                LspStoreEvent::LanguageServerAdded(..) | LspStoreEvent::LanguageServerRemoved(..)
            ) && let Some(registration) = this.projects.get_mut(&project_id)
            {
                registration.targets = lsp_targets(lsp_store.read(cx));
            }
        });
        self.projects.insert(
            project_id,
            ProjectProcesses {
                targets,
                consumers: 1,
                _lsp_subscription: subscription,
            },
        );
        self.ensure_polling(cx);
    }

    fn unregister_project(&mut self, project_id: EntityId, cx: &mut Context<Self>) {
        let should_remove = self
            .projects
            .get_mut(&project_id)
            .is_some_and(|registration| {
                registration.consumers = registration.consumers.saturating_sub(1);
                registration.consumers == 0
            });
        if should_remove {
            self.projects.remove(&project_id);
        }
        self.stop_if_unused(cx);
    }

    fn register_detail_consumer(&mut self, cx: &mut Context<Self>) {
        self.detail_consumers = self.detail_consumers.saturating_add(1);
        self.ensure_polling(cx);
    }

    fn unregister_detail_consumer(&mut self, cx: &mut Context<Self>) {
        self.detail_consumers = self.detail_consumers.saturating_sub(1);
        self.stop_if_unused(cx);
    }

    fn has_consumers(&self) -> bool {
        self.detail_consumers > 0 || !self.projects.is_empty()
    }

    fn stop_if_unused(&mut self, cx: &mut Context<Self>) {
        if !self.has_consumers() {
            self.poll_task.take();
            self.snapshots.clear();
            self.has_cpu_sample = false;
            cx.notify();
        }
    }

    fn set_enabled(&mut self, enabled: bool, cx: &mut Context<Self>) {
        if self.enabled == enabled {
            return;
        }

        self.enabled = enabled;
        if !enabled {
            self.poll_task.take();
            self.snapshots.clear();
            self.has_cpu_sample = false;
            cx.notify();
            return;
        }

        self.ensure_polling(cx);
        cx.notify();
    }

    fn ensure_polling(&mut self, cx: &mut Context<Self>) {
        if !self.enabled || !self.has_consumers() || self.poll_task.is_some() {
            return;
        }

        self.poll_task = Some(cx.spawn(async move |this, cx| {
            let mut system = System::new();
            let mut sample_count = 0usize;

            loop {
                let targets = match this.update(cx, |this, cx| this.process_targets(cx)) {
                    Ok(targets) => targets,
                    Err(_) => break,
                };

                let sample = cx
                    .background_spawn(async move { sample_processes(system, targets) })
                    .await;
                system = sample.0;
                sample_count = sample_count.saturating_add(1);

                if this
                    .update(cx, |this, cx| {
                        this.snapshots = sample.1;
                        this.has_cpu_sample = sample_count >= 2;
                        cx.notify();
                    })
                    .is_err()
                {
                    break;
                }

                let interval = match this.update(cx, |this, cx| this.poll_interval(cx)) {
                    Ok(interval) => interval,
                    Err(_) => break,
                };
                cx.background_executor().timer(interval).await;
            }
        }));
    }

    fn poll_interval(&self, cx: &mut App) -> Duration {
        if self.detail_consumers > 0 {
            return DETAIL_POLL_INTERVAL;
        }

        if cx.active_window().is_some() {
            FOREGROUND_POLL_INTERVAL
        } else {
            BACKGROUND_POLL_INTERVAL
        }
    }

    fn process_targets(&self, _cx: &App) -> Vec<ProcessTarget> {
        let mut targets = Vec::new();
        let mut seen_pids = HashSet::new();
        if let Ok(pid) = sysinfo::get_current_pid() {
            seen_pids.insert(pid.as_u32());
            targets.push(ProcessTarget {
                name: "Zed".into(),
                pid: pid.as_u32(),
                kind: ProcessKind::Zed,
            });
        }

        for project in self.projects.values() {
            targets.extend(
                project
                    .targets
                    .iter()
                    .filter(|target| seen_pids.insert(target.pid))
                    .cloned(),
            );
        }
        targets
    }
}

fn lsp_targets(lsp_store: &LspStore) -> Vec<ProcessTarget> {
    lsp_store
        .language_server_statuses()
        .filter_map(|(_, status)| {
            Some(ProcessTarget {
                name: status.name.0.clone(),
                pid: status.process_id?,
                kind: ProcessKind::LanguageServer,
            })
        })
        .collect()
}

struct GlobalResourceMetrics(Entity<ResourceMetrics>);

impl Global for GlobalResourceMetrics {}

pub struct ResourceMonitorButton {
    metrics: Entity<ResourceMetrics>,
    _metrics_subscription: Subscription,
}

impl ResourceMonitorButton {
    pub fn new(project: Entity<Project>, window: &mut Window, cx: &mut Context<Self>) -> Self {
        let metrics = ResourceMetrics::global(cx);
        let project_id = project.entity_id();
        metrics.update(cx, |metrics, cx| metrics.register_project(&project, cx));
        let weak_metrics = metrics.downgrade();
        cx.on_release(move |_, cx| {
            weak_metrics
                .update(cx, |metrics, cx| metrics.unregister_project(project_id, cx))
                .ok();
        })
        .detach();
        let metrics_subscription = cx.observe_in(&metrics, window, |_, _, _, cx| cx.notify());
        Self {
            metrics,
            _metrics_subscription: metrics_subscription,
        }
    }
}

impl StatusItemView for ResourceMonitorButton {
    fn set_active_pane_item(
        &mut self,
        _: Option<&dyn ItemHandle>,
        _: &mut Window,
        _: &mut Context<Self>,
    ) {
    }

    fn hide_setting(&self, _: &App) -> Option<workspace::HideStatusItem> {
        None
    }
}

impl Render for ResourceMonitorButton {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let metrics = self.metrics.read(cx);
        let total_memory = metrics
            .snapshots
            .iter()
            .map(|process| process.memory_bytes)
            .sum::<u64>();
        let label = if !metrics.enabled {
            "Resources Off".to_string()
        } else if metrics.snapshots.is_empty() {
            "Resources…".to_string()
        } else if metrics.has_cpu_sample {
            let total_cpu = metrics
                .snapshots
                .iter()
                .map(|process| process.cpu_percent)
                .sum::<f32>();
            format!("{total_cpu:.0}% · {}", format_memory(total_memory))
        } else {
            format_memory(total_memory)
        };

        let enabled = metrics.enabled;

        h_flex()
            .gap_0()
            .child(
                Button::new("resource-monitor-status", label)
                    .style(ButtonStyle::Subtle)
                    .label_size(LabelSize::Small)
                    .tooltip(Tooltip::text(
                        "Zed + language server CPU and memory. Click for details.",
                    ))
                    .on_click(|_, window, cx| {
                        window.dispatch_action(Box::new(OpenResourceMonitor), cx);
                    }),
            )
            .child(
                IconButton::new("toggle-resource-monitor", IconName::Power)
                    .icon_size(IconSize::XSmall)
                    .toggle_state(enabled)
                    .tooltip(Tooltip::text(if enabled {
                        "Turn Resource Monitor off"
                    } else {
                        "Turn Resource Monitor on"
                    }))
                    .on_click(|_, _, cx| {
                        let metrics = ResourceMetrics::global(cx);
                        metrics.update(cx, |metrics, cx| metrics.set_enabled(!metrics.enabled, cx));
                    }),
            )
    }
}

struct ResourceMonitor {
    metrics: Entity<ResourceMetrics>,
    focus_handle: FocusHandle,
    _metrics_subscription: Subscription,
}

impl ResourceMonitor {
    fn new(metrics: Entity<ResourceMetrics>, window: &mut Window, cx: &mut Context<Self>) -> Self {
        metrics.update(cx, |metrics, cx| metrics.register_detail_consumer(cx));
        let weak_metrics = metrics.downgrade();
        cx.on_release(move |_, cx| {
            weak_metrics
                .update(cx, |metrics, cx| metrics.unregister_detail_consumer(cx))
                .ok();
        })
        .detach();
        let metrics_subscription = cx.observe_in(&metrics, window, |_, _, _, cx| cx.notify());
        Self {
            metrics,
            focus_handle: cx.focus_handle(),
            _metrics_subscription: metrics_subscription,
        }
    }

    fn copy_report(&self, cx: &mut Context<Self>) {
        let metrics = self.metrics.read(cx);
        let mut report = String::from("Zed Resource Monitor\n");
        report.push_str("Name\tPID\tCPU\tMemory\n");
        for process in &metrics.snapshots {
            let cpu = metrics
                .has_cpu_sample
                .then(|| format!("{:.1}%", process.cpu_percent))
                .unwrap_or_else(|| "—".to_string());
            report.push_str(&format!(
                "{}\t{}\t{}\t{}\n",
                process.name,
                process.pid,
                cpu,
                format_memory(process.memory_bytes)
            ));
        }
        cx.write_to_clipboard(ClipboardItem::new_string(report));
    }

    fn render_header(&self, cx: &App) -> Div {
        h_flex()
            .px_3()
            .py_2()
            .gap_3()
            .border_b_1()
            .border_color(cx.theme().colors().border)
            .child(Label::new("Process").size(LabelSize::Small).flex_1())
            .child(metric_header("PID"))
            .child(metric_header("CPU"))
            .child(metric_header("Memory"))
    }

    fn render_section(
        &self,
        title: &'static str,
        processes: &[ProcessSnapshot],
        has_cpu_sample: bool,
        cx: &App,
    ) -> Div {
        v_flex()
            .w_full()
            .child(
                div()
                    .px_3()
                    .pt_3()
                    .pb_1()
                    .child(Label::new(title).size(LabelSize::Small).color(Color::Muted)),
            )
            .children(processes.iter().map(|process| {
                let cpu = has_cpu_sample
                    .then(|| format!("{:.1}%", process.cpu_percent))
                    .unwrap_or_else(|| "—".to_string());

                h_flex()
                    .id(("resource-process", process.pid as usize))
                    .mx_1()
                    .px_2()
                    .py_1()
                    .gap_3()
                    .rounded_sm()
                    .hover(|style| style.bg(cx.theme().colors().ghost_element_hover))
                    .child(
                        Label::new(process.name.clone())
                            .size(LabelSize::Small)
                            .flex_1()
                            .truncate(),
                    )
                    .child(metric(process.pid.to_string(), Color::Muted))
                    .child(metric(cpu, Color::Default))
                    .child(metric(format_memory(process.memory_bytes), Color::Default))
            }))
    }
}

fn sample_processes(
    mut system: System,
    targets: Vec<ProcessTarget>,
) -> (System, Vec<ProcessSnapshot>) {
    let pids = targets
        .iter()
        .map(|target| Pid::from_u32(target.pid))
        .collect::<Vec<_>>();
    let refresh = ProcessRefreshKind::nothing()
        .with_cpu()
        .with_memory()
        .without_tasks();
    system.refresh_processes_specifics(ProcessesToUpdate::Some(&pids), true, refresh);

    let mut snapshots = targets
        .into_iter()
        .filter_map(|target| {
            let process = system.process(Pid::from_u32(target.pid))?;
            Some(ProcessSnapshot {
                name: target.name,
                pid: target.pid,
                cpu_percent: process.cpu_usage(),
                memory_bytes: process.memory(),
                kind: target.kind,
            })
        })
        .collect::<Vec<_>>();
    snapshots.sort_by(|left, right| {
        left.kind
            .cmp(&right.kind)
            .then_with(|| left.name.cmp(&right.name))
            .then_with(|| left.pid.cmp(&right.pid))
    });
    (system, snapshots)
}

fn metric_header(label: &'static str) -> Div {
    div()
        .w(px(76.))
        .text_right()
        .child(Label::new(label).size(LabelSize::Small).color(Color::Muted))
}

fn metric(value: impl Into<SharedString>, color: Color) -> Div {
    div()
        .w(px(76.))
        .text_right()
        .child(Label::new(value).size(LabelSize::Small).color(color))
}

fn format_memory(bytes: u64) -> String {
    const MIB: f64 = 1024. * 1024.;
    const GIB: f64 = 1024. * MIB;
    if bytes as f64 >= GIB {
        format!("{:.1} GiB", bytes as f64 / GIB)
    } else {
        format!("{:.1} MiB", bytes as f64 / MIB)
    }
}

impl EventEmitter<()> for ResourceMonitor {}

impl Focusable for ResourceMonitor {
    fn focus_handle(&self, _: &App) -> FocusHandle {
        self.focus_handle.clone()
    }
}

impl Render for ResourceMonitor {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let metrics = self.metrics.read(cx);
        let total_cpu = metrics.has_cpu_sample.then(|| {
            metrics
                .snapshots
                .iter()
                .map(|process| process.cpu_percent)
                .sum::<f32>()
        });
        let total_memory = metrics
            .snapshots
            .iter()
            .map(|process| process.memory_bytes)
            .sum::<u64>();
        let summary = if !metrics.enabled {
            "Monitoring paused".to_string()
        } else {
            total_cpu.map_or_else(
                || {
                    format!(
                        "{} processes · {}",
                        metrics.snapshots.len(),
                        format_memory(total_memory)
                    )
                },
                |cpu| {
                    format!(
                        "{} processes · {:.1}% CPU · {}",
                        metrics.snapshots.len(),
                        cpu,
                        format_memory(total_memory)
                    )
                },
            )
        };

        let zed = metrics
            .snapshots
            .iter()
            .filter(|process| process.kind == ProcessKind::Zed)
            .cloned()
            .collect::<Vec<_>>();
        let language_servers = metrics
            .snapshots
            .iter()
            .filter(|process| process.kind == ProcessKind::LanguageServer)
            .cloned()
            .collect::<Vec<_>>();

        v_flex()
            .id("resource-monitor")
            .key_context("ResourceMonitor")
            .track_focus(&self.focus_handle)
            .size_full()
            .child(
                h_flex()
                    .px_3()
                    .py_2()
                    .gap_2()
                    .justify_between()
                    .border_b_1()
                    .border_color(cx.theme().colors().border)
                    .child(
                        v_flex()
                            .min_w_0()
                            .child(Label::new("Resource Monitor"))
                            .child(
                                Label::new(summary)
                                    .size(LabelSize::Small)
                                    .color(Color::Muted)
                                    .truncate(),
                            ),
                    )
                    .child(
                        Button::new("copy-resource-report", "Copy Report")
                            .style(ButtonStyle::Subtle)
                            .label_size(LabelSize::Small)
                            .on_click(cx.listener(|this, _, _window, cx| {
                                this.copy_report(cx);
                            })),
                    ),
            )
            .child(self.render_header(cx))
            .child(
                v_flex()
                    .id("resource-monitor-processes")
                    .size_full()
                    .overflow_scroll()
                    .when(!zed.is_empty(), |this| {
                        this.child(self.render_section("Zed", &zed, metrics.has_cpu_sample, cx))
                    })
                    .when(!language_servers.is_empty(), |this| {
                        this.child(self.render_section(
                            "Language Servers",
                            &language_servers,
                            metrics.has_cpu_sample,
                            cx,
                        ))
                    })
                    .when(!metrics.enabled, |this| {
                        this.child(
                            v_flex()
                                .size_full()
                                .items_center()
                                .justify_center()
                                .gap_1()
                                .child(Label::new("Resource monitoring is off"))
                                .child(
                                    Label::new("Turn it on from the status bar")
                                        .size(LabelSize::Small)
                                        .color(Color::Muted),
                                ),
                        )
                    })
                    .when(metrics.enabled && metrics.snapshots.is_empty(), |this| {
                        this.child(
                            v_flex()
                                .size_full()
                                .items_center()
                                .justify_center()
                                .gap_1()
                                .child(Label::new("Collecting process metrics…"))
                                .child(
                                    Label::new("CPU requires two samples")
                                        .size(LabelSize::Small)
                                        .color(Color::Muted),
                                ),
                        )
                    }),
            )
    }
}

impl Item for ResourceMonitor {
    type Event = ();

    fn to_item_events(_: &Self::Event, _: &mut dyn FnMut(workspace::item::ItemEvent)) {}

    fn tab_content_text(&self, _detail: usize, _cx: &App) -> SharedString {
        "Resource Monitor".into()
    }

    fn telemetry_event_text(&self) -> Option<&'static str> {
        None
    }

    fn clone_on_split(
        &self,
        _workspace_id: Option<WorkspaceId>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Task<Option<Entity<Self>>> {
        Task::ready(Some(
            cx.new(|cx| ResourceMonitor::new(self.metrics.clone(), window, cx)),
        ))
    }
}
