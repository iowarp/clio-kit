import React from 'react';
import Link from '@docusaurus/Link';
import {catalogue, featuredItems} from './data';
import {Frame, Glyph, Icon, Card} from './shared';
import {Catalog} from './Catalog';
import styles from './styles.module.css';
export {ComponentPage} from './Details';
export {PublishersPage} from './Publishers';

export function Home() {
  return (
    <Frame title="The scientific AI meta-marketplace">
      <div className={styles.container}>
        <section className={styles.hero}>
          <div>
            <div className={styles.eyebrow}>
              <span className={styles.dot} /> THE IOWARP META-MARKETPLACE
            </div>
            <h1>
              Scientific tools.
              <br />
              <span>Connected workflows.</span>
            </h1>
            <p className={styles.heroDescription}>
              Bring the right skills, tools and agents to your research.
              <br className={styles.desktopBreak} /> From a single dataset to
              your next HPC workflow.
            </p>
            <div className={styles.heroActions}>
              <a className={styles.primary} href="#catalogue">
                Explore the catalogue <Icon name="arrow" size={18} />
              </a>
              <Link className={styles.secondary} to="/docs/intro">
                Get started <Icon name="terminal" size={17} />
              </Link>
            </div>
            <div className={styles.heroNote}>
              Open source <span>·</span> Built by researchers <span>·</span>{' '}
              Part of IoWarp
            </div>
          </div>
          <div className={styles.researchCard}>
            <div className={styles.researchHeading}>
              <span className={styles.dot} /> BUILT AROUND YOUR RESEARCH{' '}
              <span className={styles.smallOrbit}>↗</span>
            </div>
            <div className={styles.flowRow}>
              <Glyph kind="skill" />
              <div>
                <strong>Start with a question</strong>
                <span>Skills guide the workflow</span>
              </div>
              <span className={styles.stepNumber}>01</span>
            </div>
            <div className={styles.flowConnector} />
            <div className={styles.flowRow}>
              <Glyph kind="mcp" />
              <div>
                <strong>Connect your tools</strong>
                <span>MCP servers work with your data</span>
              </div>
              <span className={styles.stepNumber}>02</span>
            </div>
            <div className={styles.flowConnector} />
            <div className={styles.flowRow}>
              <Glyph kind="agent" />
              <div>
                <strong>Review the evidence</strong>
                <span>Agents help plan and check results</span>
              </div>
              <span className={styles.stepNumber}>03</span>
            </div>
            <div className={styles.researchFooter}>
              <span>Data. Compute. Discovery.</span>
              <Icon name="workflow" size={17} />
            </div>
          </div>
        </section>

        <section
          className={styles.workflowSection}
          aria-labelledby="workflow-heading"
        >
          <div className={styles.sectionHeading}>
            <div>
              <span className={styles.eyebrow}>A GOOD PLACE TO START</span>
              <h2 id="workflow-heading">One workflow. The right tools.</h2>
            </div>
            <Link to="/?type=plugin#catalogue" className={styles.textLink}>
              Explore plugins <Icon name="arrow" size={16} />
            </Link>
          </div>
          <div className={styles.featuredGrid}>
            {featuredItems().map((item) => (
              <Card
                key={item.id}
                item={item}
                featured
              />
            ))}
          </div>
        </section>

        <Catalog />
        <section className={styles.contribute}>
          <div>
            <span className={styles.eyebrow}>
              A MARKETPLACE THAT GROWS WITH YOU
            </span>
            <h2>Your expertise belongs here.</h2>
            <p>
              Share an MCP, skill, agent or hook. Bundle them into a plugin,
              or bring your own marketplace.
            </p>
          </div>
          <Link to="/docs/authoring" className={styles.secondary}>
            Start contributing <Icon name="arrow" size={18} />
          </Link>
        </section>
        <div className={styles.credits}>
          Built at the <a href="https://grc.iit.edu/">Gnosis Research Center</a>
          , Illinois Institute of Technology.
          <br />
          Part of the IoWarp platform. Supported in part by the National Science
          Foundation.
        </div>
      </div>
    </Frame>
  );
}
