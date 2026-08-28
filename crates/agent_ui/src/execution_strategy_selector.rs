use agent::Thread;
use agent_settings::{AgentAutonomy, AgentExecutionStrategy};
use gpui::{Context, Entity, Subscription, Window, prelude::*};
use ui::{
    Button, ContextMenu, ContextMenuEntry, Icon, IconName, LabelSize, PopoverMenu,
    PopoverMenuHandle, Tooltip, prelude::*,
};

/// Native-only selector for the thread's execution policy.
///
/// This is deliberately separate from ACP session modes and native profiles:
/// profiles control permissions, while this selector controls coordination.
pub struct ExecutionStrategySelector {
    thread: Entity<Thread>,
    menu_handle: PopoverMenuHandle<ContextMenu>,
    _subscriptions: Vec<Subscription>,
}

impl ExecutionStrategySelector {
    pub fn new(thread: Entity<Thread>, cx: &mut Context<Self>) -> Self {
        let subscription = cx.observe(&thread, |_, _, cx| cx.notify());
        Self {
            thread,
            menu_handle: PopoverMenuHandle::default(),
            _subscriptions: vec![subscription],
        }
    }

    fn build_context_menu(
        &self,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Entity<ContextMenu> {
        let thread = self.thread.clone();
        ContextMenu::build(window, cx, move |mut menu, _window, _cx| {
            let current_strategy = thread.read(_cx).execution_strategy();
            let current_autonomy = thread.read(_cx).autonomy();

            menu = menu.header("Execution strategy");
            for (strategy, label) in [
                (AgentExecutionStrategy::Direct, "Direct"),
                (AgentExecutionStrategy::Plan, "Plan"),
                (AgentExecutionStrategy::Orchestrate, "Orchestrate"),
                (AgentExecutionStrategy::Auto, "Auto"),
            ] {
                let thread = thread.clone();
                menu = menu.item(
                    ContextMenuEntry::new(label)
                        .toggleable(IconPosition::End, strategy == current_strategy)
                        .handler(move |_window, cx| {
                            thread.update(cx, |thread, cx| {
                                thread.set_execution_policy(strategy, thread.autonomy(), cx);
                            });
                        }),
                );
            }

            menu = menu.separator().header("Approval policy");
            for (autonomy, label) in [
                (AgentAutonomy::Manual, "Manual"),
                (AgentAutonomy::Supervised, "Supervised"),
                (AgentAutonomy::Autonomous, "Autonomous"),
            ] {
                let thread = thread.clone();
                menu = menu.item(
                    ContextMenuEntry::new(label)
                        .toggleable(IconPosition::End, autonomy == current_autonomy)
                        .handler(move |_window, cx| {
                            thread.update(cx, |thread, cx| {
                                thread.set_execution_policy(
                                    thread.execution_strategy(),
                                    autonomy,
                                    cx,
                                );
                            });
                        }),
                );
            }
            menu
        })
    }
}

impl Render for ExecutionStrategySelector {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let thread = self.thread.read(cx);
        let strategy = thread.execution_strategy();
        let autonomy = thread.autonomy();
        let label = match strategy {
            AgentExecutionStrategy::Direct => "Direct",
            AgentExecutionStrategy::Plan => "Plan",
            AgentExecutionStrategy::Orchestrate => "Orchestrate",
            AgentExecutionStrategy::Auto => "Auto",
        };
        let tooltip = format!("Execution: {label} · Approval: {autonomy:?}");
        let this = cx.weak_entity();

        PopoverMenu::new("execution-strategy-selector")
            .trigger_with_tooltip(
                Button::new("execution-strategy-trigger", label)
                    .label_size(LabelSize::Small)
                    .color(Color::Muted)
                    .end_icon(
                        Icon::new(IconName::ChevronDown)
                            .size(IconSize::XSmall)
                            .color(Color::Muted),
                    ),
                Tooltip::text(tooltip),
            )
            .anchor(gpui::Anchor::BottomRight)
            .with_handle(self.menu_handle.clone())
            .menu(move |window, cx| {
                this.update(cx, |this, cx| this.build_context_menu(window, cx))
                    .ok()
            })
    }
}
