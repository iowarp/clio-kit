import React, {useEffect, useMemo, useState} from 'react';
import Link from '@docusaurus/Link';
import Heading from '@theme/Heading';
import {useHistory, useLocation} from '@docusaurus/router';
import {catalogue, kinds, publisherFor} from './data';
import {Icon, Card} from './shared';
import styles from './styles.module.css';

export function Catalog({fixedPublisher}) {
  const location = useLocation();
  const history = useHistory();
  const params = new URLSearchParams(location.search);
  const query = params.get('q') || '';
  const type = params.get('type') || 'all';
  const origin = params.get('origin') || 'all';
  const domain = params.get('domain') || 'all';
  const publisher = fixedPublisher || params.get('publisher') || 'all';
  const [limit, setLimit] = useState(12);
  const [moreFilters, setMoreFilters] = useState(false);
  const base = useMemo(
    () =>
      catalogue.items.filter(
        (item) => publisher === 'all' || item.publisher === publisher,
      ),
    [publisher],
  );
  const results = useMemo(
    () =>
      base
        .filter(
          (item) =>
            (type === 'all' || item.kind === type) &&
            (origin === 'all' || item.origin === origin) &&
            (domain === 'all' || item.category === domain) &&
            [
              item.title,
              item.name,
              item.description,
              item.category,
              ...item.tags,
              publisherFor(item)?.name,
            ]
              .join(' ')
              .toLowerCase()
              .includes(query.toLowerCase().trim()),
        )
        .sort((a, b) => a.title.localeCompare(b.title)),
    [base, type, origin, domain, query],
  );
  useEffect(() => setLimit(12), [location.search, fixedPublisher]);
  const update = (key, value) => {
    const next = new URLSearchParams(location.search);
    if (!value || value === 'all') next.delete(key);
    else next.set(key, value);
    history.replace({
      pathname: location.pathname,
      search: next.toString() ? `?${next}` : '',
      hash: location.hash,
    });
  };
  const clear = () =>
    history.replace({
      pathname: location.pathname,
      search: fixedPublisher
        ? `?publisher=${encodeURIComponent(fixedPublisher)}`
        : '',
      hash: '#catalogue',
    });
  return (
    <section className={styles.catalogue} aria-labelledby="catalogue">
      <div className={styles.sectionHeading}>
        <div>
          <span className={styles.eyebrow}>FIND YOUR NEXT CAPABILITY</span>
          <Heading as="h2" id="catalogue">
            Explore the catalogue
          </Heading>
        </div>
        {!fixedPublisher && (
          <Link to="/publishers" className={styles.textLink}>
            <Icon name="people" size={17} /> Meet the publishers{' '}
            <Icon name="arrow" size={16} />
          </Link>
        )}
      </div>
      <div className={styles.catalogueLayout}>
        <aside
          className={`${styles.filters} ${moreFilters ? styles.filtersExpanded : ''}`}
          aria-label="Catalogue filters"
        >
          <span className={styles.filterLabel}>COMPONENT TYPE</span>
          <div className={styles.kindFilters}>
            {kinds.map(([key, label]) => (
              <button
                key={key}
                type="button"
                className={`${styles.filterButton} ${type === key ? styles.activeFilter : ''}`}
                aria-pressed={type === key}
                onClick={() => update('type', key)}
              >
                <Icon name={key} size={17} />
                <span>{label}</span>
                <span className={styles.count}>
                  {key === 'all'
                    ? base.length
                    : base.filter((r) => r.kind === key).length}
                </span>
              </button>
            ))}
          </div>
          <button
            className={styles.moreFilters}
            aria-expanded={moreFilters}
            onClick={() => setMoreFilters(!moreFilters)}
          >
            {moreFilters ? 'Fewer filters −' : 'More filters +'}
          </button>
          <label className={styles.filterLabel} htmlFor="domain-filter">
            RESEARCH AREA
          </label>
          <select
            id="domain-filter"
            value={domain}
            onChange={(e) => update('domain', e.target.value)}
            className={styles.select}
          >
            <option value="all">All areas</option>
            {[...new Set(base.map((r) => r.category))].sort().map((c) => (
              <option key={c}>{c}</option>
            ))}
          </select>
          <label className={styles.filterLabel} htmlFor="origin-filter">
            SOURCE
          </label>
          <select
            id="origin-filter"
            value={origin}
            onChange={(e) => update('origin', e.target.value)}
            className={styles.select}
          >
            <option value="all">All sources</option>
            <option>Maintained</option>
            <option>Adapted</option>
            <option>Indexed</option>
          </select>
          {!fixedPublisher && (
            <>
              <label className={styles.filterLabel} htmlFor="publisher-filter">
                PUBLISHER
              </label>
              <select
                id="publisher-filter"
                value={publisher}
                onChange={(e) => update('publisher', e.target.value)}
                className={styles.select}
              >
                <option value="all">All publishers</option>
                {catalogue.publishers.map((p) => (
                  <option key={p.id} value={p.id}>
                    {p.name}
                  </option>
                ))}
              </select>
            </>
          )}
          <div className={styles.filterHelp}>
            <Icon name="plugin" size={20} />
            <strong>New to CLIO Kit?</strong>
            <p>
              Start with a workflow bundle, or choose just the components you
              need.
            </p>
            <Link to="/docs/intro">Installation guide ↗</Link>
          </div>
        </aside>
        <div className={styles.resultArea}>
          <div className={styles.searchBox}>
            <Icon name="search" />
            <input
              aria-label="Search catalogue"
              placeholder="Search skills, tools, workflows…"
              value={query}
              onChange={(e) => update('q', e.target.value)}
            />
            {query && (
              <button aria-label="Clear search" onClick={() => update('q', '')}>
                ×
              </button>
            )}
            <span className={styles.searchHint}>EXPLORE</span>
          </div>
          <div className={styles.resultsMeta}>
            <span role="status" aria-live="polite">
              {results.length}{' '}
              {results.length === 1 ? 'component' : 'components'}
              {query && <> matching “{query}”</>}
            </span>
            {query ||
            type !== 'all' ||
            origin !== 'all' ||
            domain !== 'all' ||
            (!fixedPublisher && publisher !== 'all') ? (
              <button className={styles.clearFilters} onClick={clear}>
                Clear filters
              </button>
            ) : (
              <span>Sorted A–Z</span>
            )}
          </div>
          {results.length ? (
            <>
              <div className={styles.cardGrid}>
                {results.slice(0, limit).map((item) => (
                  <Card key={item.id} item={item} />
                ))}
              </div>
              {limit < results.length && (
                <div className={styles.loadMore}>
                  <button
                    className={styles.secondary}
                    onClick={() => setLimit(limit + 12)}
                  >
                    Show more components{' '}
                    <span>({results.length - limit} remaining)</span>
                  </button>
                </div>
              )}
            </>
          ) : (
            <div className={styles.empty}>
              <Icon name={type === 'hook' ? 'hook' : 'search'} size={32} />
              <h3>
                {type === 'hook' && !query
                  ? 'Ready for your first hook'
                  : 'No matching components'}
              </h3>
              <p>
                {type === 'hook' && !query
                  ? 'Hook authoring and validation are supported. No standalone maintained hook package is currently listed.'
                  : 'Try a different term, research area or source.'}
              </p>
              <button className={styles.secondary} onClick={clear}>
                Reset filters
              </button>
              {type === 'hook' && (
                <Link to="/docs/authoring#add-a-hook">
                  Learn how to add a hook ↗
                </Link>
              )}
            </div>
          )}
        </div>
      </div>
    </section>
  );
}
