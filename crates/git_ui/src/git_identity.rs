use crate::git_panel::GitPanel;
use editor::Editor;
use fs::Fs;
use gpui::{
    App, AppContext, Context, DismissEvent, Entity, EventEmitter, FocusHandle, Focusable,
    InteractiveElement, SharedString, WeakEntity, Window,
};
use menu::{Cancel, Confirm};
use project::project_settings::GitCommitIdentity;
use settings::{GitCommitIdentityContent, update_settings_file};
use std::sync::Arc;
use ui::{
    Button, ButtonCommon, ButtonStyle, Clickable, Color, ContextMenu, ContextMenuEntry, Headline,
    HeadlineSize, Icon, IconName, IconPosition, IconSize, Label, LabelCommon, LabelSize,
    ParentElement, Render, Styled, StyledExt, div, h_flex, v_flex,
};
use util::ResultExt;
use workspace::ModalView;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum GitIdentitySource {
    Repository,
    Inherited,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct GitIdentity {
    pub name: SharedString,
    pub email: SharedString,
    pub source: GitIdentitySource,
}

impl GitIdentity {
    pub fn initials(&self) -> SharedString {
        let initials = self
            .name
            .split_whitespace()
            .filter_map(|part| part.chars().next())
            .take(2)
            .flat_map(char::to_uppercase)
            .collect::<String>();
        if initials.is_empty() {
            "?".into()
        } else {
            initials.into()
        }
    }

    pub fn tooltip(&self) -> SharedString {
        let source = match self.source {
            GitIdentitySource::Repository => "Repository identity",
            GitIdentitySource::Inherited => "Inherited Git identity",
        };
        format!("{} <{}>\n{}", self.name, self.email, source).into()
    }
}

pub(crate) fn parse_git_config_list(output: &str) -> Vec<(&str, &str)> {
    output
        .split('\0')
        .filter_map(|entry| {
            let (key, value) = entry.split_once('\n')?;
            Some((key, value))
        })
        .collect()
}

pub(crate) fn identity_from_config(effective: &str, local: &str) -> Option<GitIdentity> {
    let mut name = None;
    let mut email = None;
    for (key, value) in parse_git_config_list(effective) {
        match key.to_ascii_lowercase().as_str() {
            "user.name" => name = Some(value.trim()),
            "user.email" => email = Some(value.trim()),
            _ => {}
        }
    }

    let has_local_override = parse_git_config_list(local).iter().any(|(key, _)| {
        key.eq_ignore_ascii_case("user.name") || key.eq_ignore_ascii_case("user.email")
    });

    let name = name.filter(|name| !name.is_empty())?;
    let email = email.filter(|email| !email.is_empty())?;
    Some(GitIdentity {
        name: name.to_owned().into(),
        email: email.to_owned().into(),
        source: if has_local_override {
            GitIdentitySource::Repository
        } else {
            GitIdentitySource::Inherited
        },
    })
}

pub(crate) fn valid_email(email: &str) -> bool {
    let email = email.trim();
    !email.is_empty()
        && !email.chars().any(char::is_whitespace)
        && email
            .split_once('@')
            .is_some_and(|(local, domain)| !local.is_empty() && !domain.is_empty())
}

fn include_selected_repository_identity(
    mut configured_identities: Vec<GitCommitIdentity>,
    selected_identity: Option<&GitIdentity>,
) -> Vec<GitCommitIdentity> {
    if let Some(identity) =
        selected_identity.filter(|identity| identity.source == GitIdentitySource::Repository)
        && !configured_identities.iter().any(|configured| {
            configured.name == identity.name.as_ref() && configured.email == identity.email.as_ref()
        })
    {
        configured_identities.insert(
            0,
            GitCommitIdentity {
                name: identity.name.to_string(),
                email: identity.email.to_string(),
            },
        );
    }
    configured_identities
}

pub(crate) fn build_commit_identity_menu(
    mut menu: ContextMenu,
    configured_identities: Vec<GitCommitIdentity>,
    selected_identity: Option<GitIdentity>,
    git_panel: WeakEntity<GitPanel>,
) -> ContextMenu {
    let inherited_selected = selected_identity
        .as_ref()
        .is_some_and(|identity| identity.source == GitIdentitySource::Inherited);
    let inherited_panel = git_panel.clone();
    let inherited_label = selected_identity
        .as_ref()
        .filter(|identity| identity.source == GitIdentitySource::Inherited)
        .map(|identity| format!("Global: {} <{}>", identity.name, identity.email))
        .unwrap_or_else(|| "Use global Git identity".to_string());
    menu = menu.header("Commit Identity").toggleable_entry(
        inherited_label,
        inherited_selected,
        IconPosition::Start,
        None,
        move |_, cx| {
            inherited_panel
                .update(cx, |panel, cx| panel.use_inherited_commit_identity(cx))
                .log_err();
        },
    );

    let configured_identities =
        include_selected_repository_identity(configured_identities, selected_identity.as_ref());

    if configured_identities.is_empty() {
        menu = menu.item(ContextMenuEntry::new("No saved identities").disabled(true));
    }

    for configured_identity in configured_identities {
        let selected = selected_identity.as_ref().is_some_and(|identity| {
            identity.source == GitIdentitySource::Repository
                && identity.name.as_ref() == configured_identity.name
                && identity.email.as_ref() == configured_identity.email
        });
        let label = format!(
            "{} <{}>",
            configured_identity.name, configured_identity.email
        );
        let name = configured_identity.name;
        let email = configured_identity.email;
        let identity_panel = git_panel.clone();
        menu = menu.toggleable_entry(label, selected, IconPosition::Start, None, move |_, cx| {
            identity_panel
                .update(cx, |panel, cx| {
                    panel.apply_commit_identity(name.clone(), email.clone(), cx)
                })
                .log_err();
        });
    }

    menu.separator()
        .entry("Add identity…", None, move |window, cx| {
            let git_panel = git_panel.clone();
            window.defer(cx, move |window, cx| {
                git_panel
                    .update(cx, |panel, cx| {
                        panel.open_configure_commit_identity(window, cx)
                    })
                    .log_err();
            })
        })
}

pub(crate) struct ConfigureGitIdentityModal {
    name_editor: Entity<Editor>,
    email_editor: Entity<Editor>,
    error: Option<SharedString>,
    git_panel: WeakEntity<GitPanel>,
    fs: Arc<dyn Fs>,
}

impl ConfigureGitIdentityModal {
    pub(crate) fn new(
        git_panel: WeakEntity<GitPanel>,
        fs: Arc<dyn Fs>,
        current_identity: Option<GitIdentity>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Self {
        let name_editor = cx.new(|cx| {
            let mut editor = Editor::single_line(window, cx);
            editor.set_placeholder_text("Name", window, cx);
            if let Some(identity) = current_identity.as_ref() {
                editor.set_text(identity.name.to_string(), window, cx);
            }
            editor
        });
        let email_editor = cx.new(|cx| {
            let mut editor = Editor::single_line(window, cx);
            editor.set_placeholder_text("Email", window, cx);
            if let Some(identity) = current_identity.as_ref() {
                editor.set_text(identity.email.to_string(), window, cx);
            }
            editor
        });
        Self {
            name_editor,
            email_editor,
            error: None,
            git_panel,
            fs,
        }
    }

    fn cancel(&mut self, _: &Cancel, _window: &mut Window, cx: &mut Context<Self>) {
        cx.emit(DismissEvent);
    }

    fn confirm(&mut self, _: &Confirm, _window: &mut Window, cx: &mut Context<Self>) {
        let name = self.name_editor.read(cx).text(cx).trim().to_owned();
        let email = self.email_editor.read(cx).text(cx).trim().to_owned();
        if name.is_empty() {
            self.error = Some("Enter a Git author name".into());
            cx.notify();
            return;
        }
        if !valid_email(&email) {
            self.error = Some("Enter a valid email address".into());
            cx.notify();
            return;
        }

        update_settings_file(self.fs.clone(), cx, {
            let name = name.clone();
            let email = email.clone();
            move |settings, _| {
                let git = settings.git.get_or_insert_default();
                if !git
                    .commit_identities
                    .iter()
                    .any(|identity| identity.name.trim() == name && identity.email.trim() == email)
                {
                    git.commit_identities
                        .push(GitCommitIdentityContent { name, email });
                }
            }
        });

        self.git_panel
            .update(cx, |panel, cx| panel.apply_commit_identity(name, email, cx))
            .log_err();
        cx.emit(DismissEvent);
    }
}

impl EventEmitter<DismissEvent> for ConfigureGitIdentityModal {}
impl ModalView for ConfigureGitIdentityModal {}

impl Focusable for ConfigureGitIdentityModal {
    fn focus_handle(&self, cx: &App) -> FocusHandle {
        self.name_editor.focus_handle(cx)
    }
}

impl Render for ConfigureGitIdentityModal {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl gpui::IntoElement {
        v_flex()
            .key_context("ConfigureGitIdentityModal")
            .on_action(cx.listener(Self::cancel))
            .on_action(cx.listener(Self::confirm))
            .elevation_2(cx)
            .w(gpui::rems(28.))
            .p_3()
            .gap_3()
            .child(
                h_flex()
                    .gap_1p5()
                    .child(Icon::new(IconName::Person).size(IconSize::Small))
                    .child(Headline::new("Add Git Identity").size(HeadlineSize::XSmall)),
            )
            .child(
                v_flex()
                    .gap_1()
                    .child(Label::new("Name").size(LabelSize::Small))
                    .child(div().w_full().child(self.name_editor.clone())),
            )
            .child(
                v_flex()
                    .gap_1()
                    .child(Label::new("Email").size(LabelSize::Small))
                    .child(div().w_full().child(self.email_editor.clone())),
            )
            .children(self.error.clone().map(|error| {
                h_flex()
                    .gap_1()
                    .child(
                        Icon::new(IconName::XCircle)
                            .size(IconSize::Small)
                            .color(Color::Error),
                    )
                    .child(Label::new(error).size(LabelSize::Small).color(Color::Error))
            }))
            .child(
                h_flex()
                    .justify_end()
                    .gap_2()
                    .child(
                        Button::new("cancel-git-identity", "Cancel")
                            .style(ButtonStyle::Subtle)
                            .on_click(
                                cx.listener(|this, _, window, cx| this.cancel(&Cancel, window, cx)),
                            ),
                    )
                    .child(
                        Button::new("save-git-identity", "Save")
                            .style(ButtonStyle::Filled)
                            .on_click(cx.listener(|this, _, window, cx| {
                                this.confirm(&Confirm, window, cx)
                            })),
                    ),
            )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_effective_and_local_identity() {
        let effective = "user.name\nGlobal Name\0user.email\nglobal@example.com\0user.name\nLocal Name\0user.email\nlocal@example.com\0";
        let local = "core.filemode\ntrue\0user.name\nLocal Name\0user.email\nlocal@example.com\0";
        let identity = identity_from_config(effective, local).expect("identity should parse");
        assert_eq!(identity.name.as_ref(), "Local Name");
        assert_eq!(identity.email.as_ref(), "local@example.com");
        assert_eq!(identity.source, GitIdentitySource::Repository);
    }

    #[test]
    fn rejects_incomplete_identity_and_invalid_email() {
        assert!(identity_from_config("user.name\nName\0", "").is_none());
        assert!(!valid_email("not-an-email"));
        assert!(valid_email("name@example"));
        assert!(valid_email("name@example.com"));
    }

    #[test]
    fn includes_unsaved_repository_identity_in_picker() {
        let selected_identity = GitIdentity {
            name: "Local Name".into(),
            email: "local@example.com".into(),
            source: GitIdentitySource::Repository,
        };
        let configured_identities = include_selected_repository_identity(
            vec![GitCommitIdentity {
                name: "Work Name".to_string(),
                email: "work@example.com".to_string(),
            }],
            Some(&selected_identity),
        );

        assert_eq!(configured_identities.len(), 2);
        assert_eq!(configured_identities[0].name, "Local Name");
        assert_eq!(configured_identities[0].email, "local@example.com");
    }
}
