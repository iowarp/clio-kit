import catalogue from '@site/src/data/catalogue.json';
export {catalogue};

export const kinds = [
  ['all', 'All components'],
  ['workflow', 'Workflows'],
  ['skill', 'Skills'],
  ['mcp', 'MCP servers'],
  ['plugin', 'Plugins'],
  ['agent', 'Agents'],
  ['hook', 'Hooks'],
  ['service', 'Services'],
];
export const clientNames = {
  'claude-code': 'Claude Code',
  codex: 'Codex',
  antigravity: 'Antigravity',
  other: 'Other clients',
};
export const itemUrl = (item) => `/component?id=${encodeURIComponent(item.id)}`;
export const publisherUrl = (id) =>
  `/publishers?publisher=${encodeURIComponent(id)}`;
export const publisherFor = (item) =>
  catalogue.publishers.find((p) => p.id === item.publisher);
export const kindLabel = (kind) =>
  kinds.find(([key]) => key === kind)?.[1] || kind;
export const summary = (text) =>
  text
    .replace(/^Use when this workflow is requested:\s*/i, '')
    .replace(/^Use when\s+/i, '')
    .split(' Triggers on')[0];

export function installation(item, client) {
  if (item.kind === 'skill') {
    const target =
      client === 'claude-code'
        ? '.claude/skills'
        : client === 'other'
          ? '/path/to/agent/skills'
          : '.agents/skills';
    return {
      label: 'Install this skill',
      code: `clio-kit skill install ${item.name} --target ${target}`,
      note: 'Installs the procedure and its resources. Configure required MCP servers separately. Other host tools may need their own setup.',
    };
  }
  if (item.kind === 'mcp') {
    const command = `clio-kit mcp-server ${item.name}`;
    const code =
      client === 'codex'
        ? `codex mcp add clio-${item.name} -- ${command}`
        : client === 'claude-code'
          ? `claude mcp add --scope project clio-${item.name} -- ${command}`
          : JSON.stringify(
              {
                mcpServers: {
                  [`clio-${item.name}`]: {
                    command: 'clio-kit',
                    args: ['mcp-server', item.name],
                  },
                },
              },
              null,
              2,
            );
    return {
      label:
        client === 'other' || client === 'antigravity'
          ? 'MCP configuration'
          : 'Register this MCP server',
      code,
      note: 'Install the launcher first. Restart or reload your client and verify the MCP connection. System backends may require additional setup.',
    };
  }
  if (item.kind === 'service')
    return {
      label: 'Start the search service',
      code: 'clio-kit search serve',
      note: 'This is a standalone retrieval service, not an MCP server. Configure and index your document collection using the service guide.',
    };
  const plugin = item.plugin || item.name;
  return {
    label: 'From your CLIO Kit checkout',
    code: `claude plugin marketplace add "$PWD"\nclaude plugin install ${plugin}@clio-kit`,
    note:
      item.origin === 'Indexed'
        ? 'Installs the upstream package from the pinned marketplace source. Upstream code and prerequisites remain with its maintainer.'
        : 'Native plugins use Claude Code’s manifest format. Other agents can install portable skills and configure MCPs separately.',
  };
}
