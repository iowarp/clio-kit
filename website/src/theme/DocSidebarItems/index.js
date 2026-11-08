import React, {useState} from 'react';
import Link from '@docusaurus/Link';
import {usePluginData} from '@docusaurus/useGlobalData';
import DocSidebarItems from '@theme-original/DocSidebarItems';

// Clio Coder's ranking: title matches first, then section headings.
function rank(guide, query) {
  const title = guide.title.toLowerCase();
  const score =
    title === query
      ? 100
      : title.startsWith(query)
        ? 60
        : title.includes(query)
          ? 40
          : 0;
  return (
    score + (guide.headings.join(' ').toLowerCase().includes(query) ? 10 : 0)
  );
}

function Highlight({text, query}) {
  const at = text.toLowerCase().indexOf(query);
  if (at < 0) return text;
  return (
    <>
      {text.slice(0, at)}
      <mark>{text.slice(at, at + query.length)}</mark>
      {text.slice(at + query.length)}
    </>
  );
}

function FindGuide({onItemClick}) {
  const guides = usePluginData('clio-doc-search');
  const [value, setValue] = useState('');
  const query = value.trim().toLowerCase();
  const found =
    query.length < 2
      ? []
      : guides
          .filter((guide) =>
            `${guide.title} ${guide.excerpt} ${guide.headings.join(' ')}`
              .toLowerCase()
              .includes(query),
          )
          .sort((a, b) => rank(b, query) - rank(a, query))
          .slice(0, 10);
  // Arrow keys move between the box and the results; Escape clears the search.
  const onKeyDown = (event) => {
    const input = event.currentTarget.querySelector('input');
    const links = [...event.currentTarget.querySelectorAll('a')];
    if (event.key === 'Escape') {
      setValue('');
      input.focus();
    } else if (
      (event.key === 'ArrowDown' || event.key === 'ArrowUp') &&
      links.length
    ) {
      event.preventDefault();
      const next =
        links.indexOf(document.activeElement) +
        (event.key === 'ArrowDown' ? 1 : -1);
      (next < 0 ? input : links[Math.min(next, links.length - 1)]).focus();
    }
  };
  let status = '';
  if (query.length >= 2) {
    status = found.length
      ? `Found ${found.length} ${found.length === 1 ? 'guide' : 'guides'}.`
      : 'No matching guides. Try a server, command or task name.';
  }
  return (
    <li className="kit-doc-search">
      <div role="search" onKeyDown={onKeyDown}>
        <label htmlFor="doc-search">Find a guide</label>
        <input
          id="doc-search"
          type="search"
          placeholder="Search topics…"
          autoComplete="off"
          value={value}
          onChange={(event) => setValue(event.target.value)}
          aria-describedby="doc-search-status"
        />
        <p id="doc-search-status" role="status">
          {status}
        </p>
        {found.length > 0 && (
          <ul>
            {found.map((guide) => (
              <li key={guide.url}>
                <Link
                  to={guide.url}
                  onClick={() => {
                    setValue('');
                    onItemClick?.({type: 'link', href: guide.url});
                  }}
                >
                  <Highlight text={guide.title} query={query} />
                  <small>{guide.group}</small>
                </Link>
              </li>
            ))}
          </ul>
        )}
      </div>
    </li>
  );
}

// The search sits under the docs home link, as on coder.iowarp.ai.
export default function DocSidebarItemsWithSearch(props) {
  if (props.level !== 1) return <DocSidebarItems {...props} />;
  const [home, ...rest] = props.items;
  return (
    <>
      <DocSidebarItems {...props} items={[home]} />
      <FindGuide onItemClick={props.onItemClick} />
      <DocSidebarItems {...props} items={rest} />
    </>
  );
}
