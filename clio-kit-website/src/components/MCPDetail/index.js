import React, { useEffect, useState } from 'react';
import Link from '@docusaurus/Link';
import CodeBlock from '@theme/CodeBlock';
import styles from './styles.module.css';

const MCPDetail = ({ 
  name, 
  icon, 
  category, 
  description, 
  version, 
  actions, 
  platforms,
  keywords,
  license,
  tools = [],
  children 
}) => {
  const [activeTab, setActiveTab] = useState('installation');
  const [activeInstallTab, setActiveInstallTab] = useState('cursor');
  const [expandedAction, setExpandedAction] = useState(null);

  const serverName = name.toLowerCase().replace(/ /g, '-');
  useEffect(() => {
    const showLinkedSection = () => {
      const anchor = decodeURIComponent(window.location.hash.slice(1));
      if (anchor && !['installation', 'actions'].includes(anchor)) {
        setActiveTab('examples');
      }
    };
    showLinkedSection();
    window.addEventListener('hashchange', showLinkedSection);
    return () => window.removeEventListener('hashchange', showLinkedSection);
  }, []);

  const installationConfigs = {
    codex: {
      title: 'Codex',
      description: 'Registers the server in ~/.codex/config.toml for Codex CLI and its IDE extension.',
      language: 'bash',
      code: `codex mcp add ${serverName}-mcp -- clio-kit mcp-server ${serverName}
codex mcp list`
    },
    cursor: {
      title: 'Cursor',
      description: 'Merge into .cursor/mcp.json in your project, or ~/.cursor/mcp.json for user-wide access.',
      language: 'json',
      code: `{
  "mcpServers": {
    "${serverName}-mcp": {
      "command": "clio-kit",
      "args": ["mcp-server", "${serverName}"]
    }
  }
}`
    },
    vscode: {
      title: 'VS Code / Copilot',
      description: 'Merge into .vscode/mcp.json, then use MCP: List Servers to start it. The Codex extension uses the Codex tab instead.',
      language: 'json',
      code: `{
  "servers": {
    "${serverName}-mcp": {
      "type": "stdio",
      "command": "clio-kit",
      "args": ["mcp-server", "${serverName}"]
    }
  }
}`
    },
    claude_code: {
      title: 'Claude Code',
      description: 'Run from your working project. For workflow bundles, skills, and agents, see the integration guide below.',
      language: 'bash',
      code: `claude mcp add --scope project ${serverName}-mcp -- clio-kit mcp-server ${serverName}`
    },
    antigravity: {
      title: 'Antigravity',
      description: 'Open MCP Servers → Manage MCP Servers → View raw config, or use workspace .agents/mcp_config.json. Merge this entry and reload the MCP configuration.',
      language: 'json',
      code: `{
  "mcpServers": {
    "${serverName}-mcp": {
      "command": "clio-kit",
      "args": ["mcp-server", "${serverName}"]
    }
  }
}`
    },
    claude_desktop: {
      title: 'Claude Desktop',
      description: 'Open Settings → Developer → Edit Config, merge into claude_desktop_config.json, and restart the application.',
      language: 'json',
      code: `{
  "mcpServers": {
    "${serverName}-mcp": {
      "command": "clio-kit",
      "args": ["mcp-server", "${serverName}"]
    }
  }
}`
    },
    manual: {
      title: 'Manual Setup',
      description: 'Install this checkout before configuring a client. A directly launched stdio server waits for MCP input; use doctor --connect to check the connection.',
      language: 'bash',
      code: `git clone https://github.com/iowarp/clio-kit.git
cd clio-kit
uv tool install --force --reinstall ".[verification]"
clio-kit mcp-server ${serverName}`
    }
  };


  // Simple markdown-like renderer for tool descriptions
  const renderMarkdownDescription = (text) => {
    if (!text) return <p>No description available.</p>;
    
    // Split by bullet points and render as list
    const lines = text.split('\n').filter(line => line.trim());
    const hasBullets = lines.some(line => line.trim().startsWith('-') || line.trim().startsWith('*'));
    
    if (hasBullets) {
      const listItems = lines
        .filter(line => line.trim().startsWith('-') || line.trim().startsWith('*'))
        .map((line, index) => (
          <li key={index}>{line.replace(/^[\s\-\*]+/, '').trim()}</li>
        ));
      
      const nonListContent = lines
        .filter(line => !(line.trim().startsWith('-') || line.trim().startsWith('*')))
        .join(' ');
      
      return (
        <div>
          {nonListContent && <p>{nonListContent}</p>}
          {listItems.length > 0 && <ul>{listItems}</ul>}
        </div>
      );
    }
    
    return <p>{text}</p>;
  };

  const toggleAction = (actionName) => {
    setExpandedAction(expandedAction === actionName ? null : actionName);
  };

  return (
    <div className={styles.mcpDetail}>
      {/* Header */}
      <div className={styles.header}>
        <Link to="/" className={styles.backButton}>
          ← Back to MCPs
        </Link>
        
        <div className={styles.mcpInfo}>
          <div className={styles.mcpHeader}>
            <div className={styles.mcpIcon}>{icon}</div>
            <div className={styles.mcpTitleSection}>
              <h1 className={styles.mcpTitle}>{name}</h1>
              <div className={styles.mcpMeta}>
                <span className={styles.mcpCategory}>{category}</span>
                <span className={styles.mcpVersion}>v{version}</span>
              </div>
            </div>
          </div>
          <p className={styles.mcpDescription}>{description}</p>
          
          {/* Project Information */}
          {(keywords || license) && (
            <div className={styles.projectInfo}>
              {keywords && keywords.length > 0 && (
                <div className={styles.projectInfoItem}>
                  <strong>Keywords:</strong> {keywords.slice(0, 8).join(', ')}
                </div>
              )}
              {license && (
                <div className={styles.projectInfoItem}>
                  <strong>License:</strong> {license}
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Tab Navigation */}
      <div className={styles.tabNavigation}>
        <button 
          className={`${styles.mainTab} ${activeTab === 'installation' ? styles.active : ''}`}
          onClick={() => setActiveTab('installation')}
        >
          Installation
        </button>
        <button 
          className={`${styles.mainTab} ${activeTab === 'actions' ? styles.active : ''}`}
          onClick={() => setActiveTab('actions')}
        >
          Actions ({actions?.length || 0})
        </button>
        <button 
          className={`${styles.mainTab} ${activeTab === 'examples' ? styles.active : ''}`}
          onClick={() => setActiveTab('examples')}
        >
          Examples
        </button>
      </div>

      {/* Tab Content */}
      <div className={styles.tabContent}>
        {activeTab === 'installation' && (
          <div className={styles.installationTab}>
            <div className={styles.quickInstall}>
              <div className={styles.installHeader}>
                <h2 id="installation">Installation Playbooks</h2>
                <p>
                  Select a preferred environment to provision the MCP server. Install this checkout with uv tool install before adding the client configuration.
                </p>
              </div>
              <div className={styles.installTabs}>
                {Object.entries(installationConfigs).map(([key, config]) => (
                  <button
                    key={key}
                    className={`${styles.installTab} ${activeInstallTab === key ? styles.active : ''}`}
                    onClick={() => setActiveInstallTab(key)}
                  >
                    {config.title}
                  </button>
                ))}
              </div>
              <div className={styles.installContent}>
                <p>{installationConfigs[activeInstallTab].description}</p>
                <CodeBlock language={installationConfigs[activeInstallTab].language}>
                  {installationConfigs[activeInstallTab].code}
                </CodeBlock>
                <p>
                  These commands configure this MCP server. Follow the{' '}
                  <Link to="/docs/intro#agent-integrations">agent integration guide</Link>{' '}
                  to install workflow skills and their other required servers, or use
                  Claude Code's native bundles and agent plugins. Use an absolute
                  path to clio-kit if your client cannot find it.
                </p>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'actions' && (
          <div className={styles.actionsTab}>
            <div className={styles.sectionHeader}>
              <h2 id="actions">Registered Tools</h2>
              <p>
                Each tool exposes a secure capability through the Model Context Protocol. Expand a card to review documentation notes.
              </p>
            </div>
            {tools && tools.length > 0 ? (
              <div className={styles.actionsGrid}>
                {tools.map((tool, index) => (
                  <div key={index} className={`${styles.actionCard} ${expandedAction === tool.name ? styles.expanded : ''}`} onClick={() => toggleAction(tool.name)}>
                    <div className={styles.actionHeader}>
                      <code className={styles.actionName}>{tool.name}</code>
                      <span className={styles.actionToggle}>
                        {expandedAction === tool.name ? '▼' : '▶'}
                      </span>
                    </div>
                    {expandedAction === tool.name && (
                      <div className={styles.actionDescription}>
                        {renderMarkdownDescription(tool.description)}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            ) : actions && actions.length > 0 && (
              <div className={styles.actionsGrid}>
                {actions.map((action, index) => (
                  <div key={index} className={`${styles.actionCard} ${expandedAction === action ? styles.expanded : ''}`} onClick={() => toggleAction(action)}>
                    <div className={styles.actionHeader}>
                      <code className={styles.actionName}>{action}</code>
                      <span className={styles.actionToggle}>
                        {expandedAction === action ? '▼' : '▶'}
                      </span>
                    </div>
                    {expandedAction === action && (
                      <div className={styles.actionDescription}>
                        <p>Tool functionality: {action.replace('_', ' ').toLowerCase()}</p>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        <div className={styles.examplesTab} hidden={activeTab !== 'examples'}>
          <h2 id="examples">Workflows and Usage Notes</h2>
          <div className={styles.markdownContent}>{children}</div>
        </div>
      </div>

      {/* Footer */}
      <div className={styles.footer}>
        <div className={styles.footerLinks}>
          <Link href="https://github.com/iowarp/clio-kit" className={styles.footerLink}>
            📖 View on GitHub
          </Link>
          <Link href="https://github.com/iowarp/clio-kit/issues" className={styles.footerLink}>
            🐛 Report Issue
          </Link>
        </div>
        <p className={styles.footerText}>
          Part of the CLIO Kit collection - bringing AI practically to science!
        </p>
      </div>
    </div>
  );
};

export default MCPDetail;
