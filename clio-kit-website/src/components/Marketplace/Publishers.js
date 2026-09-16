import React from 'react';
import Link from '@docusaurus/Link';
import {useLocation} from '@docusaurus/router';
import {catalogue, publisherUrl} from './data';
import {Frame, Icon} from './shared';
import {Catalog} from './Catalog';
import styles from './styles.module.css';

export function PublishersPage() {
  const location = useLocation();
  const id = new URLSearchParams(location.search).get('publisher');
  const publisher = catalogue.publishers.find((p) => p.id === id);
  return (
    <Frame
      title={
        publisher ? `${publisher.name} · Publisher` : 'Meet the publishers'
      }
    >
      <div className={styles.container}>
        <Link
          to={publisher ? '/publishers' : '/#catalogue'}
          className={styles.backLink}
        >
          ← {publisher ? 'All publishers' : 'Back to catalogue'}
        </Link>
        {publisher ? (
          <>
            <section className={styles.publisherHero}>
              <div className={styles.publisherMark}>
                <Icon name="people" size={36} />
              </div>
              <div>
                <span className={styles.eyebrow}>
                  {publisher.organization} · {publisher.origin}
                </span>
                <h1>{publisher.name}</h1>
                <p>{publisher.description}</p>
                <a
                  className={styles.textLink}
                  href={publisher.repository}
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  View repository <Icon name="external" size={15} />
                </a>
              </div>
            </section>
            <Catalog fixedPublisher={publisher.id} />
          </>
        ) : (
          <>
            <header className={styles.publisherIntro}>
              <span className={styles.eyebrow}>EXPERTISE, SHARED</span>
              <h1>Meet the publishers.</h1>
              <p>
                Discover the people and projects behind your scientific toolkit.
              </p>
            </header>
            <div className={styles.publisherGrid}>
              {catalogue.publishers.map((p) => (
                <article key={p.id} className={styles.publisherCard}>
                  <div className={styles.publisherMark}>
                    <Icon name="people" size={28} />
                  </div>
                  <span className={styles.typeLabel}>{p.organization}</span>
                  <h2>
                    <Link to={publisherUrl(p.id)}>{p.name}</Link>
                  </h2>
                  <p>{p.description}</p>
                  <div className={styles.publisherStats}>
                    {catalogue.items.filter((i) => i.publisher === p.id).length}{' '}
                    components <span>·</span> {p.origin}
                  </div>
                  <Link className={styles.textLink} to={publisherUrl(p.id)}>
                    Explore collection <Icon name="arrow" size={17} />
                  </Link>
                </article>
              ))}
            </div>
            <section className={styles.contribute}>
              <div>
                <h2>Bring your collection.</h2>
                <p>
                  Publishers keep their code, dependencies and release schedule.
                </p>
              </div>
              <Link
                className={styles.secondary}
                to="/docs/authoring#index-an-external-contribution"
              >
                Contribute a collection <Icon name="arrow" size={17} />
              </Link>
            </section>
          </>
        )}
      </div>
    </Frame>
  );
}
