import React, {useEffect, useState} from 'react';
import Link from '@docusaurus/Link';
import {useLocation} from '@docusaurus/router';
import {
  catalogue,
  clientNames,
  itemUrl,
  publisherUrl,
  publisherFor,
  kindLabel,
  summary,
  installation,
} from './data';
import {Frame, Glyph, Icon} from './shared';
import styles from './styles.module.css';

function CopyCode({code, label}) {
  const [status, setStatus] = useState('');
  useEffect(() => setStatus(''), [code]);
  async function copy() {
    try {
      await navigator.clipboard.writeText(code);
      setStatus('Copied');
    } catch {
      setStatus('Copy unavailable. Select the command below.');
    }
  }
  return (
    <div className={styles.codeBox}>
      <div className={styles.codeHeading}>
        <span>{label}</span>
        <button onClick={copy} aria-label="Copy installation command">
          <Icon name={status === 'Copied' ? 'check' : 'copy'} size={15} />
          {status === 'Copied' ? 'Copied' : 'Copy'}
        </button>
      </div>
      <pre>
        <code>{code}</code>
      </pre>
      <span className={styles.copyStatus} role="status">
        {status}
      </span>
    </div>
  );
}

function DetailContent({item}) {
  const [client, setClient] = useState(
    item.kind === 'skill' ? 'codex' : 'claude-code',
  );
  const install = installation(item, client);
  const publisher = publisherFor(item);
  const members = item.members.map((id) =>
    catalogue.items.find((r) => r.id === id),
  );
  const servers = item.servers
    .map((name) =>
      catalogue.items.find((r) => r.id === `mcp/${name.replace('clio-', '')}`),
    )
    .filter(Boolean);
  return (
    <>
      <Link to={`/?type=${item.kind}#catalogue`} className={styles.backLink}>
        ← Back to catalogue
      </Link>
      <header className={styles.detailHeader}>
        <Glyph kind={item.kind} name={item.name} large />
        <div>
          <div className={styles.detailBadges}>
            <span>{kindLabel(item.kind)}</span>
            <span>{item.origin}</span>
            {item.version && <span>v{item.version}</span>}
          </div>
          <h1>{item.title}</h1>
          <p>{summary(item.description)}</p>
          <Link to={publisherUrl(item.publisher)} className={styles.textLink}>
            By {publisher?.name} <Icon name="arrow" size={15} />
          </Link>
        </div>
      </header>
      <div className={styles.detailLayout}>
        <div>
          <section className={styles.detailSection}>
            <span className={styles.eyebrow}>BUILT FOR YOUR WORK</span>
            <h2>
              {item.kind === 'workflow'
                ? 'What this workflow brings together'
                : 'About this component'}
            </h2>
            <p>{item.outcome || summary(item.description)}</p>
            {item.kind === 'skill' && (
              <p>
                Skills are procedures your agent follows. They guide tool
                selection, sequencing and interpretation; they do not install
                tools or grant access to a backend.
              </p>
            )}
            {item.kind === 'agent' && (
              <p>
                This definition runs through the native Claude Code agent
                plugin. Its instructions guide planning or review; generated
                conclusions still need evidence checking.
              </p>
            )}
            {item.origin === 'Adapted' && (
              <div className={styles.notice}>
                <strong>Adapted from Clio Coder</strong>
                <p>
                  This version has a distinct CLIO Kit name and recorded
                  provenance. Native Clio Coder tools, fleets and execution
                  gates are not installed by the skill.
                </p>
              </div>
            )}
          </section>
          {!!members.length && (
            <section className={styles.detailSection}>
              <h2>
                Included components{' '}
                <span className={styles.inlineCount}>{members.length}</span>
              </h2>
              <div className={styles.memberList}>
                {members.map((member) => (
                  <Link key={member.id} to={itemUrl(member)}>
                    <Glyph kind={member.kind} name={member.name} />
                    <div>
                      <strong>{member.title}</strong>
                      <span>{kindLabel(member.kind)}</span>
                    </div>
                    <Icon name="arrow" size={17} />
                  </Link>
                ))}
              </div>
            </section>
          )}
          {!members.length && !!servers.length && (
            <section className={styles.detailSection}>
              <h2>Related MCP servers</h2>
              <div className={styles.memberList}>
                {servers.map((server) => (
                  <Link key={server.id} to={itemUrl(server)}>
                    <Glyph kind="mcp" name={server.name} />
                    <strong>{server.title}</strong>
                    <Icon name="arrow" size={17} />
                  </Link>
                ))}
              </div>
            </section>
          )}
          <section className={styles.detailSection}>
            <h2>Requirements &amp; validation</h2>
            <p>
              {item.evidence}. This describes the recorded metadata, not a
              guarantee of every workflow or client combination.
            </p>
            <p>
              First builds can download dependencies. HPC and scientific
              backends may need system software, credentials or access to a
              configured site.
            </p>
            <div className={styles.linkRow}>
              <Link to={item.docs}>
                Read the guide <Icon name="arrow" size={16} />
              </Link>
              <Link to="/docs/marketplace#scientific-acceptance-boundaries">
                Validation coverage <Icon name="external" size={14} />
              </Link>
            </div>
          </section>
        </div>
        <aside className={styles.installPanel} aria-label="Installation">
          <div className={styles.installTitle}>
            <Icon name="terminal" />
            <h2>Make it part of your toolkit</h2>
          </div>
          <p className={styles.installIntro}>
            New here?{' '}
            <Link to="/docs/intro">Install the CLIO Kit launcher first.</Link>
          </p>
          {item.clients.length > 1 ? (
            <>
              <label htmlFor="install-client" className={styles.filterLabel}>
                YOUR AGENT
              </label>
              <select
                id="install-client"
                className={styles.select}
                value={client}
                onChange={(e) => setClient(e.target.value)}
              >
                {item.clients.map((c) => (
                  <option key={c} value={c}>
                    {clientNames[c]}
                  </option>
                ))}
              </select>
            </>
          ) : (
            <div className={styles.clientBadge}>
              {item.kind === 'service'
                ? 'Standalone service'
                : 'Claude Code native plugin'}
            </div>
          )}
          <CopyCode code={install.code} label={install.label} />
          <p className={styles.installNote}>{install.note}</p>
          <dl className={styles.facts}>
            <div>
              <dt>Publisher</dt>
              <dd>
                <Link to={publisherUrl(item.publisher)}>{publisher?.name}</Link>
              </dd>
            </div>
            <div>
              <dt>Source</dt>
              <dd>{item.origin}</dd>
            </div>
            <div>
              <dt>Licence</dt>
              <dd>{item.license}</dd>
            </div>
            {item.revision && (
              <div>
                <dt>Upstream revision</dt>
                <dd>
                  <code>{item.revision.slice(0, 12)}</code>
                </dd>
              </div>
            )}
          </dl>
          <a
            className={styles.sourceLink}
            href={item.source}
            target="_blank"
            rel="noopener noreferrer"
          >
            View source repository <Icon name="external" size={15} />
          </a>
        </aside>
      </div>
    </>
  );
}

export function ComponentPage() {
  const location = useLocation();
  const id = new URLSearchParams(location.search).get('id');
  const item = catalogue.items.find((r) => r.id === id);
  return (
    <Frame title={item?.title || 'Component details'}>
      <div className={styles.container}>
        {item ? (
          <DetailContent item={item} key={item.id} />
        ) : (
          <div className={styles.empty}>
            <h1>{id ? 'Component not found' : 'Find your next capability'}</h1>
            <p>Browse the catalogue to choose a workflow, skill or tool.</p>
            <Link className={styles.primary} to="/#catalogue">
              Explore the catalogue <Icon name="arrow" />
            </Link>
          </div>
        )}
      </div>
    </Frame>
  );
}
