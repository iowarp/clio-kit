import React from 'react';
import Link from '@docusaurus/Link';
import {Frame} from '../components/Marketplace/shared';
import {metadata as install} from '@site/../docs/tutorials/install-components.md';
import {metadata as codex} from '@site/../docs/tutorials/codex-dataset.md';
import {metadata as archive} from '@site/../docs/tutorials/archive-and-summarize.md';
import {metadata as clean} from '@site/../docs/tutorials/clean-and-plot.md';
import {metadata as claude} from '@site/../docs/tutorials/claude-analysis.md';
import {metadata as storage} from '@site/../docs/tutorials/choose-storage.md';
import {metadata as contribute} from '@site/../docs/tutorials/contribute-plugin.md';
import s from '../components/Marketplace/overview.module.css';

const tutorials = [install, storage, codex, archive, clean, claude, contribute];

export default function Tutorials() {
  return (
    <Frame title="Tutorials — Learn to use CLIO Kit">
      <div className={`${s.home} ${s.page}`}>
        <header className={s.tutorialIntro}>
          <p className={s.eyebrow}>CLIO KIT / TUTORIALS</p>
          <h1>Get a feel for CLIO Kit.</h1>
          <p className={s.lede}>
            Practical guides to scientific tools and skills. Install a
            component, follow a real terminal session, and check the result with
            the supplied data.
          </p>
        </header>
        <div className={s.tutorialGrid}>
          {tutorials.map((doc) => (
            <article key={doc.id} className={s.tutorialCard}>
              <Link to={doc.permalink} tabIndex={-1} aria-hidden="true">
                <img src={doc.frontMatter.image} alt="" loading="lazy" />
              </Link>
              <p className={s.eyebrow}>{doc.frontMatter.tutorial_category}</p>
              <h2>
                <Link to={doc.permalink}>{doc.title}</Link>
              </h2>
              <p>{doc.description}</p>
              <Link
                className={s.textLink}
                to={doc.permalink}
                aria-label={`Read tutorial: ${doc.title}`}
              >
                Read the tutorial <span>→</span>
              </Link>
            </article>
          ))}
        </div>
      </div>
    </Frame>
  );
}
