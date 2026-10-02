import React, {useEffect, useRef} from 'react';
import Link from '@docusaurus/Link';
import CodeBlock from '@theme/CodeBlock';
import Heading from '@theme/Heading';
import {Frame} from './shared';
import LogoMark from './LogoMark';
import {attachLogoJourney} from './overviewMotion';
import s from './overview.module.css';

const capabilities = [
  [
    'Scientific data',
    'Inspect HDF5, ADIOS and Parquet files. Read a bounded slice before working with a large dataset.',
    '/catalogue?domain=Scientific+data',
  ],
  [
    'Analysis and figures',
    'Clean tables, calculate statistics and plot results with Pandas, Plot and ParaView.',
    '/catalogue/workflow/clio-analysis',
  ],
  [
    'Cluster workflows',
    'Connect to software environments, Slurm jobs and performance tools on your HPC system.',
    '/catalogue/workflow/clio-hpc',
  ],
  [
    'Research and discovery',
    'Find papers, build bibliographies and discover datasets for your research.',
    '/catalogue/workflow/clio-research',
  ],
  [
    'Geoscience',
    'Work with geographic features, terrain, seismic waveforms and event catalogues.',
    '/catalogue/workflow/clio-geoscience',
  ],
  [
    'Performance and logs',
    'Investigate I/O behavior, search large logs and interpret performance measurements.',
    '/catalogue/workflow/clio-performance',
  ],
  [
    'Skills and agents',
    'Add reusable procedures, scientific planning and guidance for interpreting results.',
    '/catalogue?type=skill',
  ],
  [
    'Workflow plugins',
    'Bring related MCPs, skills and supported agents or hooks together for a task.',
    '/docs/plugins',
  ],
  [
    'Community contributions',
    'Share your own components or index an external plugin or marketplace.',
    '/docs/contributing',
  ],
];

function Chapter({number, children}) {
  return (
    <p className={s.eyebrow}>
      <span>{number} /</span> {children}
    </p>
  );
}

function StorySection({children, ...props}) {
  return (
    <section
      className={`${s.section} ${s.page} ${s.story}`}
      data-logo-chapter
      {...props}
    >
      <div className={s.storyBody} data-reveal>
        {children}
      </div>
      <aside className={s.storyVisual} aria-hidden="true">
        <LogoMark />
      </aside>
    </section>
  );
}

export function Overview() {
  const page = useRef(null);
  const travellingLogo = useRef(null);
  useEffect(() => attachLogoJourney(page.current, travellingLogo.current), []);
  useEffect(() => {
    const root = page.current;
    const reduced = window.matchMedia('(prefers-reduced-motion: reduce)');
    if (reduced.matches || !('IntersectionObserver' in window)) return;
    const observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (!entry.isIntersecting) continue;
          entry.target.dataset.reveal = 'visible';
          observer.unobserve(entry.target);
        }
      },
      {threshold: 0, rootMargin: '0px 0px -32px 0px'},
    );
    const showAll = () => {
      observer.disconnect();
      root.querySelectorAll('[data-reveal]').forEach((element) => {
        element.dataset.reveal = 'visible';
      });
    };
    const onFocus = (event) => {
      const section = event.target.closest('[data-reveal="pending"]');
      if (section) {
        section.dataset.reveal = 'visible';
        observer.unobserve(section);
      }
    };
    root.querySelectorAll('[data-reveal]').forEach((element) => {
      if (element.getBoundingClientRect().top < window.innerHeight - 32) return;
      element.dataset.reveal = 'pending';
      observer.observe(element);
    });
    reduced.addEventListener('change', showAll);
    root.addEventListener('focusin', onFocus);
    return () => {
      showAll();
      reduced.removeEventListener('change', showAll);
      root.removeEventListener('focusin', onFocus);
    };
  }, []);
  return (
    <Frame title="Scientific tools for your AI agent">
      <div className={s.home} ref={page}>
        <img
          ref={travellingLogo}
          className={s.travellingLogo}
          src="/img/iowarp_logo.png"
          width="440"
          height="440"
          alt=""
          aria-hidden="true"
          hidden
        />
        <header className={`${s.page} ${s.hero}`}>
          <div className={s.topline}>
            <p className={s.eyebrow}>
              <span className={s.dot} /> A CLIO TOOL FOR SCIENTIFIC COMPUTING
            </p>
            <a href="https://github.com/iowarp/clio-kit">
              Open source / BSD-3-Clause ↗
            </a>
          </div>
          <div className={s.heroHeading}>
            <div>
              <h1>
                Scientific tools.
                <br />
                <em>In your agent.</em>
              </h1>
              <div className={s.heroIntro}>
                <p className={s.lede}>
                  Meet CLIO Kit. A meta-marketplace for scientific computing.
                </p>
                <p>
                  Connect your AI agent to HPC resources, scientific data
                  formats and research datasets. Choose the tools, add the
                  skills, and work with your own data.
                </p>
                <div className={s.actions}>
                  <a className={s.primary} href="#start">
                    Start using CLIO Kit <span>↗</span>
                  </a>
                  <Link className={s.textLink} to="/catalogue">
                    Take a quick tour <span>→</span>
                  </Link>
                </div>
              </div>
            </div>
            <div className={s.heroVisual} data-hero-logo>
              <LogoMark hero />
            </div>
          </div>
          <div className={s.factStrip}>
            <span>Scientific MCP servers</span>
            <span>Portable skills</span>
            <span>Workflow plugins</span>
            <span>Community contributions</span>
          </div>
        </header>

        <StorySection aria-labelledby="workflow-title">
          <div className={s.sectionHeading}>
            <Chapter number="01">From a question to a checked result</Chapter>
            <h2 id="workflow-title">Work across your research workflow.</h2>
            <p>
              CLIO Kit gives your agent scientific tools and the procedures for
              using them. You choose what to connect and check what comes back.
            </p>
          </div>
          <ol className={s.workflowSteps}>
            <li>
              <span>01</span>
              <h3>Choose the tools</h3>
              <p>
                Find research, inspect data, create figures or run cluster jobs.
                Install the components your task needs.
              </p>
            </li>
            <li>
              <span>02</span>
              <h3>Add the procedure</h3>
              <p>
                A skill describes the order of operations, scientific
                assumptions and checks. A plugin brings related components
                together.
              </p>
            </li>
            <li>
              <span>03</span>
              <h3>Check the result</h3>
              <p>
                Follow the tool calls. Check source references, data coverage,
                job status or output files against the task’s expected result.
              </p>
            </li>
          </ol>
          <div className={s.firstRequest}>
            <span className={s.eyebrow}>Try a first request</span>
            <p>
              “Summarize these experiment results. Check units and missing
              values, compare the runs, and explain what the data supports.”
            </p>
            <Link className={s.textLink} to="/tutorials">
              Choose a hands-on tutorial <span>→</span>
            </Link>
          </div>
        </StorySection>

        <StorySection aria-labelledby="components-title">
          <div>
            <div className={s.sectionHeading}>
              <Chapter number="02">Two ways to build your toolkit</Chapter>
              <h2 id="components-title">
                Start with one component.
                <br />
                <em>Bring a workflow together.</em>
              </h2>
              <p>
                Choose a single capability or install a plugin for the task. The
                same launcher runs the MCP servers.
              </p>
            </div>
            <div className={s.interfaceGrid}>
              <article>
                <div className={s.componentExample}>
                  <span className={s.eyebrow}>Individual components</span>
                  <h3>One capability at a time</h3>
                  <p>
                    Choose an MCP, skill, agent or hook for the task and client
                    you use.
                  </p>
                  <Link className={s.textLink} to="/catalogue">
                    Explore components ↗
                  </Link>
                </div>
                <h3>Choose each piece</h3>
                <p>
                  Connect an MCP server, then install a skill in your agent’s
                  discovery directory. Skills supply instructions; MCPs supply
                  executable tools.
                </p>
                <CodeBlock language="bash">
                  {'clio-kit mcp-servers\nclio-kit skill list'}
                </CodeBlock>
                <Link
                  className={s.textLink}
                  to="/docs/tutorials/install-components"
                >
                  Install individual components →
                </Link>
              </article>
              <article>
                <div className={s.componentExample}>
                  <span className={s.eyebrow}>Workflow plugins</span>
                  <h3>A complete workflow</h3>
                  <p>
                    Bring tools and procedures together for analysis, cluster
                    work, geoscience or research discovery.
                  </p>
                  <Link className={s.textLink} to="/catalogue?type=plugin">
                    Explore workflow plugins ↗
                  </Link>
                </div>
                <h3>Install a working collection</h3>
                <p>
                  A workflow plugin groups the tools and procedures a task
                  needs. Native agents and hooks use the supported host’s
                  format.
                </p>
                <CodeBlock language="bash">
                  {
                    '# Example: analysis tools and skills for Codex\nclio-kit plugin install clio-analysis --client codex --project .'
                  }
                </CodeBlock>
                <Link className={s.textLink} to="/docs/plugins">
                  Choose a workflow →
                </Link>
              </article>
            </div>
          </div>
        </StorySection>

        <StorySection aria-labelledby="agent-title">
          <div className={s.sectionHeading}>
            <Chapter number="03">Your agent, your environment</Chapter>
            <h2 id="agent-title">
              Run where
              <br />
              <em>your work belongs.</em>
            </h2>
            <p>
              Keep using your preferred agent. Connect tools to your local
              project or configure them for your research environment.
            </p>
            <Link className={s.textLink} to="/docs/clients">
              Connect your agent <span>→</span>
            </Link>
          </div>
          <dl className={s.agentRoutes}>
            <div>
              <dt>In the terminal</dt>
              <dd>Codex · Claude Code · OpenCode · Clio Coder</dd>
              <dd>Project configuration, skills and real tool calls</dd>
            </div>
            <div>
              <dt>In your editor</dt>
              <dd>Cursor · Antigravity · VS Code</dd>
              <dd>Use your client’s MCP configuration and skill support</dd>
            </div>
            <div>
              <dt>Through a native marketplace</dt>
              <dd>Claude Code plugins</dd>
              <dd>Supported plugin dependencies, agents and hooks</dd>
            </div>
          </dl>
        </StorySection>

        <StorySection aria-labelledby="tools-title">
          <div className={`${s.sectionHeading} ${s.headingSplit}`}>
            <div>
              <Chapter number="04">
                Tools and knowledge for your research
              </Chapter>
              <h2 id="tools-title">A scientific working toolkit.</h2>
            </div>
            <p>
              From literature and datasets to simulations, figures and
              performance analysis.
            </p>
          </div>
          <div className={s.featureList}>
            {capabilities.map(([title, description, to], i) => (
              <Link key={title} to={to}>
                <span>{String(i + 1).padStart(2, '0')}</span>
                <h3>{title}</h3>
                <p>{description}</p>
                <span aria-hidden="true">↗</span>
              </Link>
            ))}
          </div>
          <Link className={s.textLink} to="/catalogue">
            Browse all tools, skills and plugins <span>→</span>
          </Link>
        </StorySection>

        <StorySection aria-labelledby="start">
          <div className={s.sectionHeading}>
            <Chapter number="05">From installation to a first result</Chapter>
            <Heading as="h2" id="start">
              Bring your research.
              <br />
              <em>Start with one tool.</em>
            </Heading>
            <p>
              You’ll need uv and an agent that supports MCP. Start from a CLIO
              Kit checkout, then connect the components you need.
            </p>
            <Link className={s.textLink} to="/docs/clients">
              Installation & client setup <span>→</span>
            </Link>
          </div>
          <div className={s.installSteps}>
            <div>
              <span>1 / Get CLIO Kit</span>
              <CodeBlock language="bash">
                {
                  'git clone https://github.com/iowarp/clio-kit.git\ncd clio-kit\nuv tool install --force --reinstall --editable ".[verification]"'
                }
              </CodeBlock>
            </div>
            <div>
              <span>2 / Choose your tools</span>
              <CodeBlock language="bash">{'clio-kit mcp-servers'}</CodeBlock>
              <p>
                Choose a server from the{' '}
                <Link to="/catalogue?type=mcp">catalogue</Link> and follow its
                setup and connection check. First starts build its environment;
                some tools need system software or site access.
              </p>
            </div>
            <div>
              <span>3 / Connect your workspace</span>
              <p>
                Follow your <Link to="/docs/clients">agent’s setup guide</Link>,
                then try a{' '}
                <Link to="/tutorials">tutorial with sample data</Link>. For
                released packages, see{' '}
                <Link to="/docs/installation">selective installation</Link>.
              </p>
            </div>
          </div>
        </StorySection>

        <StorySection aria-labelledby="faq-title">
          <div className={s.sectionHeading}>
            <Chapter number="06">Before you begin</Chapter>
            <h2 id="faq-title">A few practical details.</h2>
          </div>
          <div className={s.faq}>
            <details>
              <summary>Do I need every server?</summary>
              <p>
                No. Choose individual components or a workflow. Released
                distributions support selective component downloads. A source
                checkout contains the repository.{' '}
                <Link to="/docs/installation">See installation options.</Link>
              </p>
            </details>
            <details>
              <summary>Can I keep using my current agent?</summary>
              <p>
                Yes, if it supports the component you want to use. MCPs and
                portable skills have several client routes. Native plugin agents
                and hooks depend on the host.{' '}
                <Link to="/docs/clients">Check your client’s setup.</Link>
              </p>
            </details>
            <details>
              <summary>Can I contribute my own tools?</summary>
              <p>
                Add a component or plugin here, or index a package you maintain
                elsewhere. External code and releases stay with their authors.{' '}
                <Link to="/docs/contributing">
                  Choose a contribution route.
                </Link>
              </p>
            </details>
            <details>
              <summary>What does a successful connection prove?</summary>
              <p>
                It proves the server responds. Backend setup, data access and
                scientific correctness still need checking.{' '}
                <Link to="/docs/marketplace#scientific-acceptance-boundaries">
                  Read the validation guide.
                </Link>
              </p>
            </details>
          </div>
        </StorySection>

        <section
          className={`${s.section} ${s.page} ${s.project}`}
          aria-labelledby="project-title"
        >
          <div className={s.projectMark}>
            <img
              src="/img/iowarp_logo.png"
              alt="IOWarp"
              width="220"
              height="220"
              loading="lazy"
            />
          </div>
          <div>
            <p className={s.eyebrow}>Part of the IOWarp platform</p>
            <h2 id="project-title">
              Built by researchers,
              <br />
              <em>for researchers.</em>
            </h2>
            <p>
              CLIO Kit brings AI assistance to scientific computing—from
              research discovery and data analysis to visualization and HPC.
              Developed at Illinois Institute of Technology’s Gnosis Research
              Center, with support in part from the National Science Foundation.
            </p>
            <div className={s.actions}>
              <a
                className={s.textLink}
                href="https://github.com/iowarp/clio-kit"
              >
                Explore the source ↗
              </a>
              <Link className={s.textLink} to="/tutorials">
                Browse tutorials ↗
              </Link>
              <a className={s.textLink} href="https://iowarp.ai">
                Part of IOWarp ↗
              </a>
            </div>
          </div>
        </section>
      </div>
    </Frame>
  );
}
