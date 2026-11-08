import React, {useState} from 'react';
import Link from '@docusaurus/Link';
import {Redirect, useLocation} from '@docusaurus/router';
import CodeBlock from '@theme/CodeBlock';
import {
  itemsById,
  clientNames,
  itemUrl,
  publisherUrl,
  publisherFor,
  kindName,
  installation,
} from './data';
import {Frame, Glyph, Icon} from './shared';
import styles from './styles.module.css';

export function DetailContent({item}) {
  const [client, setClient] = useState(
    item.kind === 'skill' ? 'codex' : 'claude-code',
  );
  const install = installation(item, client);
  const publisher = publisherFor(item);
  const members = item.members.map((id) => itemsById.get(id)).filter(Boolean);
  const servers = item.servers
    .map((name) => itemsById.get(`mcp/${name.replace('clio-', '')}`))
    .filter(Boolean);
  const components = members.length ? members : servers;
  return (
    <>
      <Link to={`/catalogue?type=${item.kind}`} className={styles.backLink}>
        ← Catalogue
      </Link>
      <header className={styles.detailHeader}>
        <Glyph kind={item.kind} name={item.name} large />
        <div>
          <p className={styles.detailMeta}>
            {kindName(item.kind)} · {item.origin}
            {item.version && ` · v${item.version}`}
          </p>
          <h1>{item.title}</h1>
          <p>{item.summary}</p>
          <div className={styles.linkRow}>
            <Link to={publisherUrl(item.publisher)}>{publisher?.name}</Link>
            <a href={item.source}>Source ↗</a>
            <span>{item.license}</span>
          </div>
        </div>
      </header>

      <section className={styles.detailInstall} aria-labelledby="install-title">
        <div className={styles.detailToolbar}>
          <h2 id="install-title">Install</h2>
          {item.clients.length > 1 ? (
            <label>
              <span>Your agent</span>
              <select
                className={styles.select}
                value={client}
                onChange={(e) => setClient(e.target.value)}
              >
                {item.clients.map((id) => (
                  <option key={id} value={id}>
                    {clientNames[id]}
                  </option>
                ))}
              </select>
            </label>
          ) : (
            <span>Claude Code</span>
          )}
        </div>
        <CodeBlock language="bash">{install.code}</CodeBlock>
        <p className={styles.installNote}>{install.note}</p>
        <Link to="/docs/clients">Setup and prerequisites →</Link>
      </section>

      {item.outcome && <p className={styles.detailOutcome}>{item.outcome}</p>}
      {!!components.length && (
        <section className={styles.detailSection}>
          <h2>
            {members.length ? 'Included components' : 'Related MCP servers'}{' '}
            <span className={styles.inlineCount}>{components.length}</span>
          </h2>
          <div className={styles.memberList}>
            {components.map((member) => (
              <Link key={member.id} to={itemUrl(member)}>
                <Glyph kind={member.kind} name={member.name} />
                <div>
                  <strong>{member.title}</strong>
                  <span>{kindName(member.kind)}</span>
                </div>
                <Icon name="arrow" size={17} />
              </Link>
            ))}
          </div>
        </section>
      )}
      <footer className={styles.detailFooter}>
        {item.origin === 'Indexed' && (
          <p>
            Maintained externally. Check the publisher’s requirements and
            supported clients.
          </p>
        )}
        {item.origin === 'Adapted' && (
          <p>
            Adapted from Clio Coder. Its native tools and execution environment
            are not included.
          </p>
        )}
        {item.revision && (
          <p>
            Source revision <code>{item.revision.slice(0, 12)}</code>
          </p>
        )}
        <div className={styles.linkRow}>
          <Link to={item.docs}>Documentation →</Link>
          <Link to="/docs/marketplace#scientific-acceptance-boundaries">
            Validation coverage →
          </Link>
        </div>
      </footer>
    </>
  );
}

export function ComponentPage() {
  const location = useLocation();
  const id = new URLSearchParams(location.search).get('id');
  const item = itemsById.get(id);
  if (item) return <Redirect to={itemUrl(item) + location.hash} />;
  return (
    <Frame title="Component details" noindex>
      <div className={styles.detailPage}>
        <div className={styles.empty}>
          <h1>{id ? 'Component not found' : 'Choose a component'}</h1>
          <Link to="/catalogue">Browse the catalogue →</Link>
        </div>
      </div>
    </Frame>
  );
}
