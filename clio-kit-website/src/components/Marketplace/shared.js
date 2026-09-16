import React from 'react';
import Link from '@docusaurus/Link';
import Layout from '@theme/Layout';
import {itemUrl, publisherFor, kindLabel, summary} from './data';
import styles from './styles.module.css';

export function Icon({name = 'all', size = 20, ...props}) {
  const paths = {
    all: (
      <>
        <rect x="3" y="3" width="7" height="7" rx="1.5" />
        <rect x="14" y="3" width="7" height="7" rx="1.5" />
        <rect x="3" y="14" width="7" height="7" rx="1.5" />
        <rect x="14" y="14" width="7" height="7" rx="1.5" />
      </>
    ),
    workflow: (
      <>
        <rect x="3" y="3" width="7" height="6" rx="2" />
        <rect x="14" y="15" width="7" height="6" rx="2" />
        <path d="M6.5 9v6a3 3 0 0 0 3 3H14M17 15V9a3 3 0 0 0-3-3h-4" />
      </>
    ),
    skill: (
      <>
        <path d="M12 5C9 3 5 3 3 4v15c3-1 6-1 9 1 3-2 6-2 9-1V4c-2-1-6-1-9 1Zm0 0v15" />
        <path d="m6 8 3 1m6 0 3-1" />
      </>
    ),
    mcp: (
      <>
        <rect x="3" y="3" width="18" height="7" rx="2" />
        <rect x="3" y="14" width="18" height="7" rx="2" />
        <path d="M7 6.5h.01M7 17.5h.01M12 7h5m-5 10h5" />
      </>
    ),
    plugin: (
      <>
        <path d="m12 3 9 5-9 5-9-5 9-5Zm-9 5v10l9 5 9-5V8M12 13v10M7 5.8l10 5.4" />
      </>
    ),
    agent: (
      <>
        <path d="m12 2 2.8 7.2L22 12l-7.2 2.8L12 22l-2.8-7.2L2 12l7.2-2.8L12 2Z" />
        <path d="m20 2 .5 1.5L22 4l-1.5.5L20 6l-.5-1.5L18 4l1.5-.5Z" />
      </>
    ),
    hook: <path d="m13 2-9 12h7l-1 8 10-13h-7l1-7Z" />,
    service: (
      <>
        <path d="M2 12h5l3-8 4 16 3-8h5" />
      </>
    ),
    search: (
      <>
        <circle cx="10.5" cy="10.5" r="6.5" />
        <path d="m16 16 5 5" />
      </>
    ),
    arrow: <path d="M5 12h14m-5-5 5 5-5 5" />,
    external: (
      <>
        <path d="M14 3h7v7m0-7L10 14M10 4H4v16h16v-6" />
      </>
    ),
    copy: (
      <>
        <rect x="8" y="8" width="12" height="13" rx="2" />
        <path d="M15 8V3H3v12h5" />
      </>
    ),
    check: <path d="m5 12 4 4L19 6" />,
    people: (
      <>
        <circle cx="9" cy="7" r="3" />
        <path d="M3 21v-3a6 6 0 0 1 12 0v3m2-18a3 3 0 0 1 0 6m1 5a5 5 0 0 1 3 5v2" />
      </>
    ),
    terminal: (
      <>
        <rect x="2" y="4" width="20" height="16" rx="3" />
        <path d="m6 9 3 3-3 3m7 0h4" />
      </>
    ),
  };
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.65"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      {...props}
    >
      {paths[name] || paths.all}
    </svg>
  );
}

export function Frame({title, children}) {
  return (
    <Layout
      title={title}
      description="Discover scientific MCP servers, skills, workflow plugins and agent tools in the IoWarp meta-marketplace."
      wrapperClassName={styles.shell}
    >
      <main className={styles.page}>{children}</main>
    </Layout>
  );
}

export function Glyph({kind, large = false}) {
  return (
    <span
      className={`${styles.glyph} ${styles[kind] || ''} ${large ? styles.largeGlyph : ''}`}
    >
      <Icon name={kind} size={large ? 30 : 21} />
    </span>
  );
}

export function Card({item, featured = false}) {
  const publisher = publisherFor(item);
  return (
    <article
      className={`${styles.card} ${featured ? styles.featuredCard : ''}`}
      data-kind={item.kind}
    >
      <div className={styles.cardTop}>
        <Glyph kind={item.kind} />
        <span className={styles.typeLabel}>{kindLabel(item.kind)}</span>
        <Icon name="arrow" size={17} />
      </div>
      <h3>
        <Link to={itemUrl(item)} className={styles.cardLink}>
          {item.title}
        </Link>
      </h3>
      <p className={styles.cardDescription}>{summary(item.description)}</p>
      {featured && (
        <div className={styles.memberChips}>
          {item.servers.slice(0, 4).map((name) => (
            <span key={name}>{name.replace('clio-', '')}</span>
          ))}
          {item.servers.length > 4 && <span>+{item.servers.length - 4}</span>}
        </div>
      )}
      <div className={styles.cardFoot}>
        <span>{publisher?.name}</span>
        <span className={styles.origin}>{item.origin}</span>
      </div>
    </article>
  );
}
