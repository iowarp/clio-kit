import React from 'react';
import Link from '@docusaurus/Link';
import {useDoc} from '@docusaurus/plugin-content-docs/client';
import s from './styles.module.css';

/**
 * Clio Coder's tutorial opening: back link, category and reading time, title,
 * summary, a fact row and the cover capture. Content comes from front matter.
 */
export default function TutorialHeader() {
  const {metadata, frontMatter: fm} = useDoc();
  const facts = [
    ['Written for', fm.written_for],
    ['Works in', fm.works_in],
    ['Basis', fm.basis],
  ].filter(([, value]) => value);
  return (
    <header className={s.header}>
      <Link className={s.back} to="/tutorials">
        <span aria-hidden="true">←</span> All tutorials
      </Link>
      <p className={s.eyebrow}>
        <span>{fm.tutorial_category}</span> {fm.tutorial_time} read
      </p>
      <h1>{metadata.title}</h1>
      <p className={s.lede}>{metadata.description}</p>
      {facts.length > 0 && (
        <dl className={s.facts}>
          {facts.map(([label, value]) => (
            <div key={label}>
              <dt>{label}</dt>
              <dd>{value}</dd>
            </div>
          ))}
        </dl>
      )}
      {fm.cover && (
        <figure className={s.cover}>
          <a href={fm.cover} aria-label="Open the full-size cover screenshot">
            <img src={fm.cover} alt={fm.cover_alt || ''} />
          </a>
          <figcaption>
            <span>{fm.cover_caption}</span>
            {fm.cover_tag && <span className={s.tag}>{fm.cover_tag}</span>}
          </figcaption>
        </figure>
      )}
    </header>
  );
}
