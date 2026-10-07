import React, {useEffect, useRef} from 'react';
import Link from '@docusaurus/Link';
import CodeBlock from '@theme/CodeBlock';
import Heading from '@theme/Heading';
import TabItem from '@theme/TabItem';
import Tabs from '@theme/Tabs';
import {Frame} from './shared';
import {catalogue} from './data';
import CopyCommand from './CopyCommand';
import LogoMark from './LogoMark';
import {attachLogoJourney} from './overviewMotion';
import s from './overview.module.css';

// The README's install routes, each as one line.
const installOptions = [
  ['release', 'Release', "uv tool install 'clio-kit>=2.11.0'"],
  [
    'once',
    'Run once',
    "uvx --from 'clio-kit>=2.11.0' clio-kit mcp-servers",
    'Runs the launcher without installing it.',
  ],
  [
    'npx',
    'npx',
    'npx skills@1.5.25 add iowarp/clio-kit --skill dataset-explore --agent codex --copy',
    <>
      Skills only, with Node.js 22.20 or newer. Use <code>--skill '*'</code> for
      every skill.
    </>,
  ],
  [
    'source',
    'Source',
    'git clone https://github.com/iowarp/clio-kit.git && cd clio-kit && uv tool install --editable .',
  ],
  [
    'agent',
    'Agent',
    'git clone https://github.com/iowarp/clio-kit.git && cd clio-kit',
    'Then ask your agent: “Read setup.md and set up CLIO Kit for me.”',
  ],
];

// Every component type, so the install steps cover more than MCP servers.
const componentOptions = [
  [
    'plugin',
    'Plugin',
    'clio-kit plugin install clio-scientific-io --client codex --project .',
    <>
      A workflow plugin writes its MCP configuration and copies its skills. For
      other agents, use <code>--client</code> with claude-code, opencode,
      cursor, antigravity or vscode.
    </>,
  ],
  [
    'mcp',
    'MCP server',
    'clio-kit mcp-servers\nclio-kit mcp-server hdf5',
    <>
      List the servers, then add the <code>clio-kit mcp-server</code> command to
      your agent’s MCP settings. The first start builds the server’s
      environment; some servers need system software or site access.
    </>,
  ],
  [
    'skill',
    'Skill',
    'clio-kit skill list\nclio-kit skill install dataset-explore --target .agents/skills',
    <>
      Use your agent’s skill folder, such as <code>.claude/skills</code> for
      Claude Code. Give <code>--bundle clio-scientific-io</code> instead of a
      name to install a workflow’s skills.
    </>,
  ],
  [
    'claude',
    'Claude Code',
    'clio-kit plugin fetch clio-dataset-report --target ./clio-selected\nclaude plugin marketplace add "$PWD/clio-selected"\nclaude plugin install clio-dataset-report@clio-kit',
    'Native Claude Code plugins can also include agents and hooks.',
  ],
];

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
    'Bring related MCP servers, skills and supported agents or hooks together for a task.',
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
    <Frame
      title="Scientific tools for your AI agent"
      description="CLIO Kit is the meta-marketplace for scientific computing: one catalogue of MCP servers, skills and workflow plugins from CLIO Kit and other publishers, for Claude Code, Codex, Clio Coder, OpenCode and other agents."
      entities={[
        {
          '@type': 'SoftwareSourceCode',
          name: 'CLIO Kit',
          codeRepository: 'https://github.com/iowarp/clio-kit',
          programmingLanguage: 'Python',
          license: 'https://github.com/iowarp/clio-kit/blob/main/LICENSE',
          publisher: {'@id': 'https://toolkit.iowarp.ai/#org'},
        },
      ]}
    >
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
            <a className={s.release} href="https://github.com/iowarp/clio-kit">
              Open source <span aria-hidden="true">/</span> v{catalogue.version}{' '}
              <span aria-hidden="true">↗</span>
            </a>
            <span className={s.tag}>New: meta-marketplace</span>
          </div>
          <div className={s.heroHeading}>
            <div>
              <h1>
                Scientific tools.
                <br />
                <em>In your agent.</em>
              </h1>
              <p className={s.lede}>
                CLIO Kit is the meta-marketplace for scientific computing. Add
                one MCP server, a skill or a whole workflow to Claude Code,
                Codex, Clio Coder or OpenCode, then check the results against
                your data.
              </p>
              <div className={s.heroTry}>
                <p className={s.eyebrow}>Install in one line</p>
                <Tabs>
                  {installOptions.map(([value, label, command, note]) => (
                    <TabItem
                      key={value}
                      value={value}
                      label={label}
                      default={value === 'release'}
                    >
                      <CopyCommand
                        command={command}
                        label={`Copy the ${label.toLowerCase()} command`}
                      />
                      {note && <p className={s.installNote}>{note}</p>}
                    </TabItem>
                  ))}
                </Tabs>
                <p className={s.installNote}>
                  Need uv?{' '}
                  <code>curl -LsSf https://astral.sh/uv/install.sh | sh</code>
                </p>
                <div className={s.heroLinks}>
                  <Link className={s.textLink} to="/docs/installation">
                    Other ways to install <span>→</span>
                  </Link>
                  <Link className={s.textLink} to="/tutorials">
                    Watch the tutorials <span>→</span>
                  </Link>
                </div>
              </div>
            </div>
            <div className={s.heroVisual} data-hero-logo>
              <LogoMark hero />
            </div>
          </div>
          <nav className={s.chapterNav} aria-label="Sections">
            <a href="#workflow">Workflow</a>
            <a href="#components">Components</a>
            <a href="#agents">Agents</a>
            <a href="#toolkit">Toolkit</a>
            <a href="#start">Install</a>
            <Link to="/catalogue">Catalogue ↗</Link>
          </nav>
        </header>

        <StorySection id="workflow" aria-labelledby="workflow-title">
          <div className={s.sectionHeading}>
            <Chapter number="01">From a question to a checked result</Chapter>
            <h2 id="workflow-title">
              Ask a scientific question.
              <br />
              <em>Check what comes back.</em>
            </h2>
            <p>
              CLIO Kit gives your agent scientific tools and the procedures for
              using them. You choose what to connect, and every tool call stays
              visible for review.
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
          <figure className={s.capture}>
            <div className={s.stageLabel}>
              <span>
                <span className={s.dot} aria-hidden="true" /> CLIO Kit in Clio
                Coder / clio-hdf5
              </span>
              <a className={s.primary} href="https://coder.iowarp.ai">
                Try Clio Coder <span aria-hidden="true">↗</span>
              </a>
            </div>
            <a
              href="/img/tutorials/hdf5-clio-tools.png"
              aria-label="Open the full-size Clio Coder session screenshot"
            >
              <img
                src="/img/tutorials/hdf5-clio-tools.png"
                width="2872"
                height="1380"
                loading="lazy"
                alt="Clio Coder loads the dataset-explore skill and calls clio-hdf5 open_file, visit, get_shape, get_dtype, list_attributes, read_partial_dataset and close_file"
              />
            </a>
            <figcaption>
              A real session in Clio Coder, IOWarp’s open-source coding agent:
              the HDF5 tutorial’s skill checks structure and units before
              reading three values.
            </figcaption>
          </figure>
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

        <StorySection id="components" aria-labelledby="components-title">
          <div className={s.sectionHeading}>
            <Chapter number="02">Two ways to build your toolkit</Chapter>
            <h2 id="components-title">
              Start with one component.
              <br />
              <em>Bring a workflow together.</em>
            </h2>
            <p>
              Choose a single capability or install a plugin for the task. The
              same launcher runs every MCP server.
            </p>
          </div>
          <div className={s.interfaceGrid}>
            <article>
              <div className={s.interfaceTitle}>
                <h3>One component</h3>
                <span className={s.tag}>MCP · skill</span>
              </div>
              <p>
                Connect an MCP server, then copy a skill into your agent’s skill
                folder. Skills supply instructions; MCP servers supply tools.
              </p>
              <CopyCommand command="clio-kit skill install dataset-explore --target .agents/skills" />
              <Link className={s.textLink} to="/docs/tutorials/install-components">
                Install individual components <span>→</span>
              </Link>
            </article>
            <article>
              <div className={s.interfaceTitle}>
                <h3>A workflow plugin</h3>
                <span className={s.tag}>Bundle</span>
              </div>
              <p>
                A workflow plugin groups the servers and skills a task needs. One
                command writes the MCP configuration and copies the skills.
              </p>
              <CopyCommand command="clio-kit plugin install clio-scientific-io --client codex --project ." />
              <Link className={s.textLink} to="/catalogue?type=plugin">
                Choose a workflow <span>→</span>
              </Link>
            </article>
          </div>
        </StorySection>

        <StorySection id="agents" aria-labelledby="agent-title">
          <div className={s.sectionHeading}>
            <Chapter number="03">Your agent, your environment</Chapter>
            <h2 id="agent-title">
              Use the agent
              <br />
              <em>you already have.</em>
            </h2>
            <p>
              Components install into your project in each client’s own format.
              Every hands-on tutorial is recorded in four of them.
            </p>
            <Link className={s.textLink} to="/docs/clients">
              Connect your agent <span>→</span>
            </Link>
          </div>
          <dl className={s.agentRoutes}>
            <div>
              <dt>In the terminal</dt>
              <dd>Claude Code · Codex · Clio Coder · OpenCode</dd>
              <dd>Project MCP configuration, skills and visible tool calls</dd>
            </div>
            <div>
              <dt>In your editor</dt>
              <dd>Cursor · Antigravity · VS Code</dd>
              <dd>Your client’s MCP configuration and skill support</dd>
            </div>
            <div>
              <dt>Through a native marketplace</dt>
              <dd>Claude Code plugins</dd>
              <dd>Supported plugin dependencies, agents and hooks</dd>
            </div>
          </dl>
          <div className={`${s.firstRequest} ${s.coderPromo}`}>
            <span className={s.eyebrow}>Also from IOWarp</span>
            <p>
              Need an agent? Clio Coder is IOWarp’s open-source coding agent for
              scientific software. Every tutorial here includes a Clio Coder
              session.
            </p>
            <a className={s.textLink} href="https://coder.iowarp.ai">
              Try Clio Coder <span>↗</span>
            </a>
          </div>
        </StorySection>

        <StorySection id="toolkit" aria-labelledby="tools-title">
          <div className={s.sectionHeading}>
            <Chapter number="04">Tools and knowledge for your research</Chapter>
            <h2 id="tools-title">
              A scientific
              <br />
              <em>working toolkit.</em>
            </h2>
            <p>
              Servers and skills for papers, datasets, cluster jobs, figures and
              I/O analysis.
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
            <Chapter number="05">Install</Chapter>
            <Heading as="h2" id="start">
              Install CLIO Kit
              <br />
              <em>and add what you need.</em>
            </Heading>
            <p>
              You need uv and an agent. Add a workflow plugin, a single MCP
              server or a skill; each downloads only what it needs.
            </p>
            <div className={s.heroLinks}>
              <Link className={s.textLink} to="/docs/installation">
                Release and selective install <span>→</span>
              </Link>
              <Link className={s.textLink} to="/docs/clients">
                Client setup <span>→</span>
              </Link>
            </div>
          </div>
          <div className={s.installSteps}>
            <div>
              <span>1 / Get CLIO Kit</span>
              <Tabs>
                <TabItem value="release" label="Release" default>
                  <CodeBlock language="bash">
                    {"uv tool install 'clio-kit>=2.11.0'"}
                  </CodeBlock>
                </TabItem>
                <TabItem value="source" label="From source">
                  <CodeBlock language="bash">
                    {
                      'git clone https://github.com/iowarp/clio-kit.git\ncd clio-kit\nuv tool install --force --reinstall --editable ".[verification]"'
                    }
                  </CodeBlock>
                </TabItem>
              </Tabs>
            </div>
            <div>
              <span>2 / Add a component</span>
              <Tabs>
                {componentOptions.map(([value, label, command, note]) => (
                  <TabItem
                    key={value}
                    value={value}
                    label={label}
                    default={value === 'plugin'}
                  >
                    <CodeBlock language="bash">{command}</CodeBlock>
                    <p>{note}</p>
                  </TabItem>
                ))}
              </Tabs>
              <p>
                Find names in the <Link to="/catalogue">catalogue</Link>.
              </p>
            </div>
            <div>
              <span>3 / Connect your agent</span>
              <p>
                Follow your <Link to="/docs/clients">agent’s setup guide</Link>,
                then try a{' '}
                <Link to="/tutorials">tutorial with sample data</Link>.
              </p>
            </div>
          </div>
        </StorySection>

        <StorySection aria-labelledby="faq-title">
          <div className={s.sectionHeading}>
            <Chapter number="06">Questions</Chapter>
            <h2 id="faq-title">Common questions.</h2>
          </div>
          <div className={s.faq}>
            <details>
              <summary>Do I need every server?</summary>
              <p>
                No. Choose individual components or a workflow. Released
                distributions download only the components you select.{' '}
                <Link to="/docs/installation">See installation options.</Link>
              </p>
            </details>
            <details>
              <summary>Can I keep using my current agent?</summary>
              <p>
                Yes, if it supports the component you want to use. MCP servers
                and portable skills work in Claude Code, Codex, Clio Coder,
                OpenCode and other clients. Native plugin agents and hooks depend
                on the host.{' '}
                <Link to="/docs/clients">Check your client’s setup.</Link>
              </p>
            </details>
            <details>
              <summary>What does a meta-marketplace index?</summary>
              <p>
                CLIO Kit’s own components and plugins from other publishers’
                repositories and marketplaces, in one catalogue. Indexed code and
                releases stay with their authors.{' '}
                <Link to="/publishers">Meet the publishers.</Link>
              </p>
            </details>
            <details>
              <summary>Can I contribute my own tools?</summary>
              <p>
                Add a component or plugin here, or index a package you maintain
                elsewhere.{' '}
                <Link to="/docs/tutorials/contribute-plugin">
                  Build, test and contribute a plugin.
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
            <h2 id="project-title">Meet the CLIO team.</h2>
            <p>Try a workflow on data you know and inspect the result.</p>
            <p className={s.credit}>
              Built by researchers, for researchers, at Illinois Institute of
              Technology with NSF support.
            </p>
            <div className={s.actions}>
              <a className={s.textLink} href="https://github.com/iowarp/clio-kit">
                Explore the source <span>↗</span>
              </a>
              <Link className={s.textLink} to="/tutorials">
                Browse tutorials <span>↗</span>
              </Link>
              <a className={s.textLink} href="https://coder.iowarp.ai">
                Try Clio Coder <span>↗</span>
              </a>
              <a className={s.textLink} href="https://iowarp.ai">
                Part of IOWarp <span>↗</span>
              </a>
            </div>
          </div>
        </section>

        <section
          className={`${s.section} ${s.page} ${s.finale}`}
          aria-labelledby="finale-title"
        >
          <h2 id="finale-title">
            Try it on a dataset
            <br />
            <em>you already know.</em>
          </h2>
          <div className={s.finaleActions}>
            <a className={s.primary} href="#start">
              Install CLIO Kit <span aria-hidden="true">↗</span>
            </a>
            <Link className={s.textLink} to="/catalogue">
              Browse the catalogue <span>→</span>
            </Link>
          </div>
        </section>
      </div>
    </Frame>
  );
}
