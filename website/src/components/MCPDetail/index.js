import React, {useEffect, useState} from 'react';
import Link from '@docusaurus/Link';
import Heading from '@theme/Heading';
import CodeBlock from '@theme/CodeBlock';
import {itemsById, clientNames, installation} from '../Marketplace/data';
import styles from './styles.module.css';

export default function MCPDetail({name, tools = [], children}) {
  const [client, setClient] = useState('claude-code');
  const [section, setSection] = useState('installation');
  useEffect(() => {
    const followAnchor = () => {
      const hash = window.location.hash.slice(1);
      setSection(
        !hash || hash === 'installation'
          ? 'installation'
          : hash === 'actions'
            ? 'actions'
            : 'examples',
      );
    };
    followAnchor();
    window.addEventListener('hashchange', followAnchor);
    return () => window.removeEventListener('hashchange', followAnchor);
  }, []);
  const serverName = name.toLowerCase().replace(/ /g, '-');
  const item = itemsById.get(`mcp/${serverName}`);
  if (!item) throw new Error(`Missing catalogue entry for MCP: ${serverName}`);
  const install = installation(item, client);
  const readme = `${item.source}/blob/main/${item.path}/README.md`;

  return (
    <div className={styles.reference}>
      <Link to="/catalogue?type=mcp">← Catalogue</Link>
      <p className={styles.meta}>
        <span aria-hidden="true">{item.icon}</span> {item.category} · v
        {item.version} · {item.license}
      </p>
      <p>{item.description}</p>
      <nav className={styles.sections} aria-label="MCP reference sections">
        {[
          ['installation', 'Install'],
          ['actions', `Tools (${tools.length})`],
          ['examples', 'Examples'],
        ].map(([id, label]) => (
          <a
            key={id}
            href={`#${id}`}
            aria-current={section === id ? 'location' : undefined}
            onClick={() => setSection(id)}
          >
            {label}
          </a>
        ))}
      </nav>

      <section
        aria-labelledby="installation"
        hidden={section !== 'installation'}
      >
        <div className={styles.toolbar}>
          <Heading as="h2" id="installation">
            Install
          </Heading>
          <label>
            <span>Your agent</span>
            <select
              value={client}
              onChange={(event) => setClient(event.target.value)}
            >
              {item.clients.map((id) => (
                <option key={id} value={id}>
                  {clientNames[id]}
                </option>
              ))}
            </select>
          </label>
        </div>
        <CodeBlock
          language={['codex', 'claude-code'].includes(client) ? 'bash' : 'json'}
        >
          {install.code}
        </CodeBlock>
        <p className={styles.note}>
          <Link to="/docs/clients#1-install-the-launcher">
            Install the launcher first.
          </Link>{' '}
          Restart your client after setup.{' '}
          <Link to="/docs/clients">Configuration locations →</Link>
        </p>
        <details className={styles.connection}>
          <summary>Check the connection</summary>
          <CodeBlock language="bash">{`clio-kit doctor --server ${serverName} --connect`}</CodeBlock>
          <p className={styles.note}>
            Checks prerequisites and tool discovery. Test on known input to
            verify results. Native backends may require additional setup.
          </p>
        </details>
      </section>

      <section aria-labelledby="actions" hidden={section !== 'actions'}>
        <Heading as="h2" id="actions">
          Tools
        </Heading>
        <dl className={styles.tools}>
          {tools.map((tool) => {
            const [brief, ...notes] = tool.description.split('\n');
            return (
              <div key={tool.name}>
                <dt>
                  <code>{tool.name}</code>
                </dt>
                <dd>
                  <p>{brief}</p>
                  {notes.join('').trim() && (
                    <details>
                      <summary>Details</summary>
                      <div className={styles.toolNotes}>
                        {notes.join('\n').trim()}
                      </div>
                    </details>
                  )}
                </dd>
              </div>
            );
          })}
        </dl>
      </section>

      <section aria-labelledby="examples" hidden={section !== 'examples'}>
        <Heading as="h2" id="examples">
          Examples
        </Heading>
        <p className={styles.note}>
          Replace example paths with your own files and check the returned
          results.
        </p>
        {children}
      </section>
      <footer className={styles.links}>
        <a href={readme}>README &amp; prerequisites ↗</a>
        <Link to="/tutorials">Tutorials →</Link>
        <Link to="/docs/marketplace#scientific-acceptance-boundaries">
          Tested coverage →
        </Link>
      </footer>
    </div>
  );
}
