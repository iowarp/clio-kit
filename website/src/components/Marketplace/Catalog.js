import React, {useEffect, useMemo, useState} from 'react';
import Link from '@docusaurus/Link';
import Heading from '@theme/Heading';
import {useHistory, useLocation} from '@docusaurus/router';
import {catalogue, hasAdaptedCopy, kinds, publisherFor} from './data';
import {Icon, Card} from './shared';
import styles from './styles.module.css';

const kindOrder = kinds.map(([key]) => key);

export function Catalog({fixedPublisher}) {
  const location = useLocation();
  const history = useHistory();
  const params = new URLSearchParams(location.search);
  const query = params.get('q') || '';
  // Preserve bookmarks from before workflows were grouped under Plugins.
  const requestedType = params.get('type') || 'all';
  const type = requestedType === 'workflow' ? 'plugin' : requestedType;
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
            // "All" lists an adapted skill once; its upstream stays under Packages.
            (type === 'all' ? !hasAdaptedCopy(item) : item.kind === type) &&
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
        // Workflow plugins first, then components, then indexed packages.
        .sort(
          (a, b) =>
            kindOrder.indexOf(a.kind) - kindOrder.indexOf(b.kind) ||
            a.title.localeCompare(b.title),
        ),
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
      <Heading
        as="h2"
        id="catalogue"
        className={fixedPublisher ? styles.catalogueHeading : styles.visuallyHidden}
      >
        {fixedPublisher ? 'Published entries' : 'Catalogue entries'}
      </Heading>
      <div className={styles.catalogueLayout}>
        <aside
          className={`${styles.filters} ${moreFilters ? styles.filtersExpanded : ''}`}
          aria-label="Catalogue filters"
        >
          <span className={styles.filterLabel}>Type</span>
          <div className={styles.kindFilters}>
            {kinds.map(([key, label]) => (
              <button
                key={key}
                type="button"
                className={`${styles.filterButton} ${type === key ? styles.activeFilter : ''}`}
                aria-pressed={type === key}
                onClick={() => update('type', key)}
              >
                <span>{label}</span>
                <span className={styles.count}>
                  {key === 'all'
                    ? base.filter((r) => !hasAdaptedCopy(r)).length
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
            Research area
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
            Source
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
                Publisher
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
          </div>
          <div className={styles.resultsMeta}>
            <span role="status" aria-live="polite">
              {results.length}{' '}
              {results.length === 1 ? 'entry' : 'entries'}
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
              <span>Plugins first, then A–Z</span>
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
                    Show more entries{' '}
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
                  ? 'No hook packages listed yet'
                  : 'No matching entries'}
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
