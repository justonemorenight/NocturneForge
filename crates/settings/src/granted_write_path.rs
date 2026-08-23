use std::path::{Path, PathBuf};

/// A sandbox writable-path grant paired with the canonical target resolved at approval time.
///
/// Keeping both paths prevents a saved permission from following a symlink to a different target
/// when the permission is reused later.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GrantedWritePath {
    /// The path shown to the user when permission was requested.
    pub requested: PathBuf,
    /// The canonical target captured when permission was approved.
    ///
    /// Legacy and manually entered settings use `None` and are resolved when enforced.
    pub resolved: Option<PathBuf>,
    /// Whether the target is on a Windows-hosted filesystem when sandboxing through WSL.
    pub on_windows_fs: bool,
}

impl GrantedWritePath {
    /// Creates a legacy-style grant that will be resolved when enforced.
    pub fn from_requested(requested: PathBuf) -> Self {
        Self {
            requested,
            resolved: None,
            on_windows_fs: false,
        }
    }

    /// Creates a grant with the canonical target captured at approval time.
    pub fn resolved(requested: PathBuf, resolved: PathBuf) -> Self {
        Self::resolved_on_fs(requested, resolved, false)
    }

    /// Creates a resolved grant and records whether its target is Windows-hosted.
    pub fn resolved_on_fs(requested: PathBuf, resolved: PathBuf, on_windows_fs: bool) -> Self {
        Self {
            requested,
            resolved: Some(resolved),
            on_windows_fs,
        }
    }

    /// Returns the canonical target when known, otherwise the requested path.
    pub fn canonical_or_requested(&self) -> &Path {
        self.resolved.as_deref().unwrap_or(&self.requested)
    }
}

impl serde::Serialize for GrantedWritePath {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match &self.resolved {
            None => self.requested.serialize(serializer),
            Some(resolved) => {
                use serde::ser::SerializeStruct as _;
                let field_count = if self.on_windows_fs { 3 } else { 2 };
                let mut state = serializer.serialize_struct("GrantedWritePath", field_count)?;
                state.serialize_field("requested", &self.requested)?;
                state.serialize_field("resolved", resolved)?;
                if self.on_windows_fs {
                    state.serialize_field("on_windows_fs", &self.on_windows_fs)?;
                }
                state.end()
            }
        }
    }
}

impl<'de> serde::Deserialize<'de> for GrantedWritePath {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(serde::Deserialize)]
        struct Object {
            requested: PathBuf,
            #[serde(default)]
            resolved: Option<PathBuf>,
            #[serde(default)]
            on_windows_fs: bool,
        }

        #[derive(serde::Deserialize)]
        #[serde(untagged)]
        enum StringOrObject {
            String(PathBuf),
            Object(Object),
        }

        Ok(match StringOrObject::deserialize(deserializer)? {
            StringOrObject::String(requested) => Self::from_requested(requested),
            StringOrObject::Object(Object {
                requested,
                resolved,
                on_windows_fs,
            }) => Self {
                requested,
                resolved,
                on_windows_fs,
            },
        })
    }
}
