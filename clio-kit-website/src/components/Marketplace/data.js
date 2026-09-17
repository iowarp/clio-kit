import catalogue from '@site/src/data/catalogue.json';
export {catalogue};

export const kinds = [
  ['all', 'All entries'],
  ['plugin', 'Plugins'],
  ['skill', 'Skills'],
  ['mcp', 'MCP servers'],
  ['agent', 'Agents'],
  ['hook', 'Hooks'],
  ['collection', 'Component collections'],
  ['package', 'External packages'],
  ['service', 'Services'],
];
export const clientNames = {
  'claude-code': 'Claude Code',
  codex: 'Codex',
  antigravity: 'Antigravity',
  opencode: 'OpenCode',
  cursor: 'Cursor',
  vscode: 'VS Code / Copilot',
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
  if (item.installation === 'portable-skill') {
    const target =
      client === 'claude-code'
        ? '.claude/skills'
        : client === 'opencode'
          ? '.opencode/skills'
          : client === 'cursor'
            ? '.cursor/skills'
            : client === 'vscode'
              ? '.github/skills'
              : client === 'other'
                ? '/path/to/agent/skills'
                : '.agents/skills';
    return {
      label: 'Install this skill',
      code: `clio-kit skill install ${item.name} --target ${target}`,
      note: 'Installs the procedure and its resources. Configure required MCP servers separately. Other host tools may need their own setup.',
    };
  }
  if (item.installation === 'launcher') {
    const command = `clio-kit mcp-server ${item.name}`;
    const code =
      client === 'codex'
        ? `codex mcp add clio-${item.name} -- ${command}`
        : client === 'claude-code'
          ? `claude mcp add --scope project clio-${item.name} -- ${command}`
          : JSON.stringify(
              client === 'opencode'
                ? {
                    mcp: {
                      [`clio-${item.name}`]: {
                        type: 'local',
                        command: ['clio-kit', 'mcp-server', item.name],
                      enabled: true,
                      timeout: 120000,
                      },
                    },
                  }
                : {
                    [client === 'vscode' ? 'servers' : 'mcpServers']: {
                      [`clio-${item.name}`]: {
                        ...(client === 'vscode' ? {type: 'stdio'} : {}),
                        command: 'clio-kit',
                        args: ['mcp-server', item.name],
                      },
                    },
                  },
              null,
              2,
            );
    return {
      label: !['codex', 'claude-code'].includes(client)
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
  const plugin = item.nativePackage || item.plugin || item.name;
  if (item.projectInstall && client !== 'claude-code') {
    const partial = item.componentTypes.some(
      (type) => !['skill', 'mcp'].includes(type),
    );
    return {
      label: 'Install selected components',
      code: `clio-kit plugin install ${plugin} --client ${client} --project /path/to/project${partial ? ' --components-only' : ''}`,
      note: partial
        ? 'Installs only the skills and MCP configuration. Native agents, hooks and commands are not adapted; this is a partial workflow installation.'
        : 'Installs the included skills and MCP configuration in your project. Reload the client, trust the project when prompted, and verify connections. This is project setup, not a native client plugin installation.',
    };
  }
  return {
    label: item.plugin
      ? `Install containing package: ${plugin}`
      : 'Download this native package',
    code: item.origin === 'Indexed'
      ? `claude plugin marketplace add iowarp/clio-kit\nclaude plugin install ${plugin}@clio-kit`
      : `clio-kit plugin fetch ${plugin} --target /path/to/clio-selected\nclaude plugin marketplace add /path/to/clio-selected\nclaude plugin install ${plugin}@clio-kit`,
    note:
      item.origin === 'Indexed'
        ? 'Installs the upstream package from the pinned marketplace source. Upstream code and prerequisites remain with its maintainer.'
        : item.plugin
          ? `This installs ${plugin} and its included components. It does not install only this ${item.kind}. Native agents and hooks require Claude Code.`
          : 'Requires a release with component artifacts published. Downloads this package and its dependencies; MCP implementations download on first launch. For unreleased checkout testing, register the checkout instead.',
  };
}
