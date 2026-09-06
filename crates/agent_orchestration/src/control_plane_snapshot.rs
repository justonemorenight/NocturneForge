use super::*;
use collections::HashSet;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AgentMailboxSnapshot {
    pub recipient: AgentPath,
    pub messages: Vec<AgentMessage>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AgentControlPlaneSnapshot {
    pub identities: Vec<AgentIdentity>,
    pub mailboxes: Vec<AgentMailboxSnapshot>,
    pub next_message_sequence: u64,
}

impl AgentControlPlane {
    pub fn snapshot(&self) -> AgentControlPlaneSnapshot {
        let state = self.inner.state.read();
        let mut identities = state.identities.values().cloned().collect::<Vec<_>>();
        identities.sort_by(|left, right| left.path.cmp(&right.path));
        let mut mailboxes = state
            .mailboxes
            .iter()
            .map(|(recipient, mailbox)| AgentMailboxSnapshot {
                recipient: recipient.clone(),
                messages: mailbox.queue.lock().messages.iter().cloned().collect(),
            })
            .collect::<Vec<_>>();
        mailboxes.sort_by(|left, right| left.recipient.cmp(&right.recipient));
        AgentControlPlaneSnapshot {
            identities,
            mailboxes,
            next_message_sequence: self.inner.next_message_sequence.load(Ordering::Acquire),
        }
    }

    pub fn restore(
        snapshot: AgentControlPlaneSnapshot,
        config: AgentControlPlaneConfig,
    ) -> Result<Self> {
        if snapshot.identities.len() > config.max_registered_agents {
            bail!(
                "agent snapshot contains {} identities, exceeding the {} entry limit",
                snapshot.identities.len(),
                config.max_registered_agents
            );
        }
        let execution_limiter = AgentExecutionLimiter::new(config.execution_limiter.clone())?;
        let mut identities = HashMap::default();
        let mut task_paths = HashMap::default();
        for identity in snapshot.identities {
            validate_identity(&identity)?;
            if identities
                .insert(identity.path.clone(), identity.clone())
                .is_some()
            {
                bail!("agent snapshot contains duplicate path '{}'", identity.path);
            }
            if let Some(task_id) = &identity.task_id
                && task_paths
                    .insert(task_id.clone(), identity.path.clone())
                    .is_some()
            {
                bail!("agent snapshot contains duplicate task id '{task_id}'");
            }
        }
        let root = identities
            .get(&AgentPath::root())
            .ok_or_else(|| anyhow::anyhow!("agent snapshot is missing the root identity"))?;
        if root.parent.is_some() || root.task_id.is_some() {
            bail!("agent snapshot root identity is invalid");
        }
        for identity in identities
            .values()
            .filter(|identity| identity.parent.is_some())
        {
            let Some(parent) = identity.parent.as_ref() else {
                continue;
            };
            if !identities.contains_key(parent) {
                bail!(
                    "agent '{}' references unknown parent '{parent}'",
                    identity.path
                );
            }
            if identity.path.parent().as_ref() != Some(parent) {
                bail!(
                    "agent '{}' is not a direct child of '{parent}'",
                    identity.path
                );
            }
        }

        let mut mailboxes = HashMap::default();
        let mut maximum_sequence = 0_u64;
        let mut message_sequences = HashSet::default();
        for mailbox_snapshot in snapshot.mailboxes {
            if !identities.contains_key(&mailbox_snapshot.recipient) {
                bail!(
                    "agent snapshot contains a mailbox for unknown agent '{}'",
                    mailbox_snapshot.recipient
                );
            }
            if mailbox_snapshot.messages.len() > config.max_messages_per_agent {
                bail!(
                    "mailbox for '{}' exceeds the {} message limit",
                    mailbox_snapshot.recipient,
                    config.max_messages_per_agent
                );
            }
            let mailbox = AgentMailbox::new();
            let mut queue = mailbox.queue.lock();
            let mut previous_sequence = 0_u64;
            for message in mailbox_snapshot.messages {
                validate_message(&message, &mailbox_snapshot.recipient, &identities, &config)?;
                if message.sequence <= previous_sequence
                    || !message_sequences.insert(message.sequence)
                {
                    bail!("agent snapshot contains duplicate or unordered message sequences");
                }
                queue.bytes = queue.bytes.saturating_add(message.body.len());
                previous_sequence = message.sequence;
                maximum_sequence = maximum_sequence.max(message.sequence);
                queue.messages.push_back(message);
            }
            if queue.bytes > config.max_mailbox_bytes_per_agent {
                bail!(
                    "mailbox for '{}' exceeds the {} byte limit",
                    mailbox_snapshot.recipient,
                    config.max_mailbox_bytes_per_agent
                );
            }
            let has_messages = !queue.messages.is_empty();
            drop(queue);
            if has_messages {
                match mailbox.activity_sender.try_send(()) {
                    Ok(()) | Err(async_channel::TrySendError::Full(())) => {}
                    Err(async_channel::TrySendError::Closed(())) => {
                        bail!("mailbox activity channel unexpectedly closed")
                    }
                }
            }
            if mailboxes
                .insert(mailbox_snapshot.recipient.clone(), mailbox)
                .is_some()
            {
                bail!(
                    "agent snapshot contains duplicate mailbox '{}'",
                    mailbox_snapshot.recipient
                );
            }
        }
        if maximum_sequence == u64::MAX {
            bail!("agent snapshot message sequence is exhausted");
        }
        for path in identities.keys() {
            mailboxes
                .entry(path.clone())
                .or_insert_with(AgentMailbox::new);
        }
        let next_message_sequence = snapshot
            .next_message_sequence
            .max(maximum_sequence.saturating_add(1))
            .max(1);

        Ok(Self {
            inner: Arc::new(AgentControlPlaneInner {
                config,
                state: RwLock::new(AgentControlPlaneState {
                    identities,
                    task_paths,
                    mailboxes,
                }),
                next_message_sequence: AtomicU64::new(next_message_sequence),
                runtime_events: RwLock::new(None),
                execution_limiter,
            }),
        })
    }
}

fn validate_identity(identity: &AgentIdentity) -> Result<()> {
    AgentPath::parse(identity.path.as_str())?;
    if identity.path != AgentPath::root() && identity.task_id.is_none() {
        bail!("non-root agent '{}' is missing its task id", identity.path);
    }
    if identity.path != AgentPath::root() && identity.target.is_none() {
        bail!(
            "non-root agent '{}' is missing its worker target",
            identity.path
        );
    }
    Ok(())
}

fn validate_message(
    message: &AgentMessage,
    recipient: &AgentPath,
    identities: &HashMap<AgentPath, AgentIdentity>,
    config: &AgentControlPlaneConfig,
) -> Result<()> {
    if &message.recipient != recipient {
        bail!("mailbox message recipient does not match its mailbox");
    }
    if message.author == message.recipient {
        bail!("agent snapshot contains a self-addressed message");
    }
    if !identities.contains_key(&message.author) || !identities.contains_key(&message.recipient) {
        bail!("agent snapshot message references an unknown agent");
    }
    if message.body.trim().is_empty() || message.body.len() > config.max_message_bytes {
        bail!("agent snapshot contains an invalid message body");
    }
    Ok(())
}
