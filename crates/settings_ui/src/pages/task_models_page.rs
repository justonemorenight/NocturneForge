use agent_settings::{
    AgentSettings, NativeSubagentRoleSettings, NativeSubagentRolesSettings,
    SubagentFallbackModelSettings,
};
use gpui::{ReadGlobal as _, ScrollHandle, prelude::*};
use language_model::LanguageModelRegistry;
use settings::{
    LanguageModelProviderSetting, LanguageModelSelection, NativeSubagentRoleContent,
    NativeSubagentRolesContent, Settings as _, SettingsStore, SubagentFallbackModelContent,
    SubagentFallbackStrategy,
};
use ui::{ContextMenu, PopoverMenu, SwitchField, ToggleState, prelude::*};

use crate::{SettingsWindow, components::SettingsSectionHeader};

#[derive(Clone, Copy)]
enum TaskRole {
    Explorer,
    FlowReader,
    CodingWorker,
}

impl TaskRole {
    fn id(self) -> &'static str {
        match self {
            Self::Explorer => "explorer",
            Self::FlowReader => "flow-reader",
            Self::CodingWorker => "coding-worker",
        }
    }

    fn title(self) -> &'static str {
        match self {
            Self::Explorer => "Explore",
            Self::FlowReader => "Flow Reader",
            Self::CodingWorker => "Coding Worker",
        }
    }

    fn description(self) -> &'static str {
        match self {
            Self::Explorer => "Searches and maps the codebase with read-only tools.",
            Self::FlowReader => "Traces behavior and dependencies before implementation.",
            Self::CodingWorker => "Implements scoped changes with write-capable tools.",
        }
    }

    fn settings(self, settings: &NativeSubagentRolesSettings) -> &NativeSubagentRoleSettings {
        match self {
            Self::Explorer => &settings.explorer,
            Self::FlowReader => &settings.flow_reader,
            Self::CodingWorker => &settings.coding_worker,
        }
    }

    fn content(
        self,
        settings: &mut NativeSubagentRolesContent,
    ) -> &mut Option<NativeSubagentRoleContent> {
        match self {
            Self::Explorer => &mut settings.explorer,
            Self::FlowReader => &mut settings.flow_reader,
            Self::CodingWorker => &mut settings.coding_worker,
        }
    }
}

#[derive(Clone)]
struct AvailableModel {
    provider_id: String,
    provider_name: SharedString,
    id: String,
    name: SharedString,
    efforts: Vec<(SharedString, SharedString)>,
    default_effort: Option<String>,
}

pub(crate) fn render_task_models_page(
    _settings_window: &SettingsWindow,
    scroll_handle: &ScrollHandle,
    _window: &mut Window,
    cx: &mut Context<SettingsWindow>,
) -> AnyElement {
    let settings = AgentSettings::get_global(cx).native_subagent_roles.clone();
    let roles_enabled = settings.enabled;
    let models = available_native_models(cx);

    v_flex()
        .id("task-models-page")
        .size_full()
        .pt_2p5()
        .px_8()
        .pb_16()
        .gap_6()
        .overflow_y_scroll()
        .track_scroll(scroll_handle)
        .child(
            SwitchField::new(
                "native-task-model-routing",
                Some("Use Role-Specific Task Models"),
                Some(
                    "Choose Native Agent models independently of the parent conversation provider. External ACP agents manage their own model selection."
                        .into(),
                ),
                roles_enabled,
                move |state, _, cx| {
                    set_roles_enabled(*state == ToggleState::Selected, cx);
                },
            )
            .tab_index(0),
        )
        .child(SettingsSectionHeader::new("Native Task Models").no_padding(true))
        .child(
            Label::new("Models from authenticated Native providers")
                .size(LabelSize::Small)
                .color(Color::Muted),
        )
        .when(models.is_empty(), |this| {
            this.child(
                v_flex()
                    .p_4()
                    .gap_1()
                    .rounded_md()
                    .border_1()
                    .border_color(cx.theme().colors().border_variant)
                    .child(Label::new("No available Native Agent models"))
                    .child(
                        Label::new(
                            "Configure a provider under LLM Providers. Only authenticated, enabled models with tool support are shown here.",
                        )
                        .size(LabelSize::Small)
                        .color(Color::Muted),
                    ),
            )
        })
        .children(
            [TaskRole::Explorer, TaskRole::FlowReader, TaskRole::CodingWorker]
                .into_iter()
                .map(|role| render_role_card(role, &settings, &models, !roles_enabled, cx)),
        )
        .child(
            Label::new(
                "Fallback runs at most once and only for provider availability, rate limits, transient network failures, or context overflow. It never masks tool, permission, verification, or cancellation failures.",
            )
            .size(LabelSize::Small)
            .color(Color::Muted),
        )
        .into_any_element()
}

fn available_native_models(cx: &App) -> Vec<AvailableModel> {
    let registry = LanguageModelRegistry::read_global(cx);
    let mut models: Vec<_> = registry
        .visible_providers()
        .into_iter()
        .filter(|provider| provider.is_authenticated(cx))
        .flat_map(|provider| {
            let provider_id = provider.id().0.to_string();
            let provider_name = provider.name().0;
            provider
                .provided_models(cx)
                .into_iter()
                .filter(|model| model.is_disabled().is_none() && model.supports_tools())
                .map(move |model| AvailableModel {
                    provider_id: provider_id.clone(),
                    provider_name: provider_name.clone(),
                    id: model.id().0.to_string(),
                    name: model.name().0,
                    efforts: model
                        .supported_effort_levels()
                        .into_iter()
                        .map(|effort| (effort.value, effort.name))
                        .collect(),
                    default_effort: model
                        .default_effort_level()
                        .map(|effort| effort.value.to_string()),
                })
        })
        .collect();
    models.sort_by(|left, right| {
        left.provider_name
            .cmp(&right.provider_name)
            .then_with(|| left.name.cmp(&right.name))
    });
    models
}

fn render_role_card(
    role: TaskRole,
    settings: &NativeSubagentRolesSettings,
    models: &[AvailableModel],
    disabled: bool,
    cx: &mut App,
) -> AnyElement {
    let role_settings = role.settings(settings);
    let primary_label = models
        .iter()
        .find(|model| {
            model.provider_id == role_settings.provider.0 && model.id == role_settings.model
        })
        .map(available_model_label)
        .unwrap_or_else(|| {
            format!("{} · {}", role_settings.provider.0, role_settings.model).into()
        });
    let selected_model = models.iter().find(|model| {
        model.provider_id == role_settings.provider.0 && model.id == role_settings.model
    });
    let effort_label = selected_model
        .and_then(|model| {
            if model.efforts.is_empty() {
                return Some(SharedString::from("Not supported"));
            }
            model
                .efforts
                .iter()
                .find(|(value, _)| value.as_ref() == role_settings.effort)
                .or_else(|| {
                    model.default_effort.as_deref().and_then(|default| {
                        model
                            .efforts
                            .iter()
                            .find(|(value, _)| value.as_ref() == default)
                    })
                })
                .map(|(_, name)| name.clone())
        })
        .unwrap_or_else(|| role_settings.effort.clone().into());
    let fallback_label: SharedString = match &role_settings.fallback {
        SubagentFallbackModelSettings::None => "None".into(),
        SubagentFallbackModelSettings::InheritFromParent => "Inherit from parent".into(),
        SubagentFallbackModelSettings::Model(selection) => models
            .iter()
            .find(|model| model.provider_id == selection.provider.0 && model.id == selection.model)
            .map(available_model_label)
            .unwrap_or_else(|| format!("{} · {}", selection.provider.0, selection.model).into()),
    };

    v_flex()
        .p_4()
        .gap_3()
        .rounded_md()
        .border_1()
        .border_color(cx.theme().colors().border_variant)
        .child(
            v_flex().gap_0p5().child(Label::new(role.title())).child(
                Label::new(role.description())
                    .size(LabelSize::Small)
                    .color(Color::Muted),
            ),
        )
        .child(render_setting_row(
            "Primary model",
            model_menu(role, "primary", primary_label, models.to_vec(), disabled),
        ))
        .child(render_setting_row(
            "Reasoning effort",
            effort_menu(
                role,
                effort_label,
                selected_model
                    .map(|model| model.efforts.clone())
                    .unwrap_or_default(),
                disabled,
            ),
        ))
        .child(render_setting_row(
            "Fallback model",
            fallback_menu(
                role,
                fallback_label,
                models.to_vec(),
                role_settings.provider.0.clone(),
                role_settings.model.clone(),
                disabled,
            ),
        ))
        .into_any_element()
}

fn available_model_label(model: &AvailableModel) -> SharedString {
    format!("{} · {}", model.provider_name, model.name).into()
}

fn render_setting_row(label: &'static str, control: impl IntoElement) -> impl IntoElement {
    h_flex()
        .min_w_0()
        .justify_between()
        .gap_4()
        .child(Label::new(label).size(LabelSize::Small))
        .child(control)
}

fn menu_button(id: String, label: SharedString, disabled: bool) -> Button {
    Button::new(id, label)
        .tab_index(0_isize)
        .style(ButtonStyle::Outlined)
        .size(ButtonSize::Medium)
        .disabled(disabled)
        .end_icon(Icon::new(IconName::ChevronDown).size(IconSize::Small))
}

fn model_menu(
    role: TaskRole,
    kind: &'static str,
    label: SharedString,
    models: Vec<AvailableModel>,
    disabled: bool,
) -> impl IntoElement {
    PopoverMenu::new(format!("{}-{kind}-model", role.id()))
        .trigger(menu_button(
            format!("{}-{kind}-model-trigger", role.id()),
            label,
            disabled || models.is_empty(),
        ))
        .menu(move |window, cx| {
            let models = models.clone();
            Some(ContextMenu::build(window, cx, move |mut menu, _, _| {
                for model in models.clone() {
                    let provider_id = model.provider_id.clone();
                    let model_id = model.id.clone();
                    let default_effort = model.default_effort.clone();
                    menu = menu.entry(available_model_label(&model), None, move |_, cx| {
                        set_primary_model(
                            role,
                            provider_id.clone(),
                            model_id.clone(),
                            default_effort.clone(),
                            cx,
                        );
                    });
                }
                menu
            }))
        })
        .anchor(gpui::Anchor::TopRight)
}

fn effort_menu(
    role: TaskRole,
    label: SharedString,
    efforts: Vec<(SharedString, SharedString)>,
    disabled: bool,
) -> impl IntoElement {
    PopoverMenu::new(format!("{}-effort", role.id()))
        .trigger(menu_button(
            format!("{}-effort-trigger", role.id()),
            label,
            disabled || efforts.is_empty(),
        ))
        .menu(move |window, cx| {
            let efforts = efforts.clone();
            Some(ContextMenu::build(window, cx, move |mut menu, _, _| {
                for (value, name) in efforts.clone() {
                    menu = menu.entry(name, None, move |_, cx| {
                        set_role_effort(role, value.to_string(), cx);
                    });
                }
                menu
            }))
        })
        .anchor(gpui::Anchor::TopRight)
}

fn fallback_menu(
    role: TaskRole,
    label: SharedString,
    models: Vec<AvailableModel>,
    primary_provider: String,
    primary_model: String,
    disabled: bool,
) -> impl IntoElement {
    PopoverMenu::new(format!("{}-fallback", role.id()))
        .trigger(menu_button(
            format!("{}-fallback-trigger", role.id()),
            label,
            disabled,
        ))
        .menu(move |window, cx| {
            let models = models.clone();
            let primary_provider = primary_provider.clone();
            let primary_model = primary_model.clone();
            Some(ContextMenu::build(window, cx, move |menu, _, _| {
                let mut menu = menu
                    .entry("None", None, move |_, cx| {
                        set_fallback(
                            role,
                            SubagentFallbackModelContent::Strategy(SubagentFallbackStrategy::None),
                            cx,
                        );
                    })
                    .entry("Inherit from parent", None, move |_, cx| {
                        set_fallback(
                            role,
                            SubagentFallbackModelContent::Strategy(
                                SubagentFallbackStrategy::InheritFromParent,
                            ),
                            cx,
                        );
                    })
                    .separator();
                for model in models.clone().into_iter().filter(|model| {
                    model.provider_id != primary_provider || model.id != primary_model
                }) {
                    let provider_id = model.provider_id.clone();
                    let model_id = model.id.clone();
                    let effort = model.default_effort.clone();
                    menu = menu.entry(available_model_label(&model), None, move |_, cx| {
                        set_fallback(
                            role,
                            SubagentFallbackModelContent::Model(LanguageModelSelection {
                                provider: LanguageModelProviderSetting(provider_id.clone()),
                                model: model_id.clone(),
                                enable_thinking: effort.is_some(),
                                effort: effort.clone(),
                                speed: None,
                            }),
                            cx,
                        );
                    });
                }
                menu
            }))
        })
        .anchor(gpui::Anchor::TopRight)
}

fn update_role(
    role: TaskRole,
    cx: &mut App,
    update: impl FnOnce(&mut NativeSubagentRoleContent) + Send + 'static,
) {
    SettingsStore::global(cx).update_settings_file(<dyn fs::Fs>::global(cx), move |settings, _| {
        let roles = settings
            .agent
            .get_or_insert_default()
            .native_subagent_roles
            .get_or_insert_default();
        update(role.content(roles).get_or_insert_default());
    });
}

fn set_roles_enabled(enabled: bool, cx: &mut App) {
    SettingsStore::global(cx).update_settings_file(<dyn fs::Fs>::global(cx), move |settings, _| {
        settings
            .agent
            .get_or_insert_default()
            .native_subagent_roles
            .get_or_insert_default()
            .enabled = Some(enabled);
    });
}

fn set_primary_model(
    role: TaskRole,
    provider: String,
    model: String,
    default_effort: Option<String>,
    cx: &mut App,
) {
    update_role(role, cx, move |settings| {
        settings.provider = Some(LanguageModelProviderSetting(provider));
        settings.model = Some(model);
        if let Some(default_effort) = default_effort {
            settings.effort = Some(default_effort);
        }
    });
}

fn set_role_effort(role: TaskRole, effort: String, cx: &mut App) {
    update_role(role, cx, move |settings| settings.effort = Some(effort));
}

fn set_fallback(role: TaskRole, fallback: SubagentFallbackModelContent, cx: &mut App) {
    update_role(role, cx, move |settings| settings.fallback = Some(fallback));
}
