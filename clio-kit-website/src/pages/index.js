import React from 'react';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import Link from '@docusaurus/Link';
import MCPShowcase from '@site/src/components/MCPShowcase';

export default function Home() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={`${siteConfig.title}`}
      description="CLIO Kit - A meta-marketplace for scientific MCP servers, skills, plugins, agents, and community contributions">
      <main className="landingPage">

        {/* Hero */}
        <section className="hero">
          <div className="hero__content">
            {/* Gnosis box at top */}
            <div className="hero__eyebrow">
              <img src="/img/logos/grc-logo.png" alt="GRC" className="hero__eyebrowLogo" />
              <a href="https://grc.iit.edu/" target="_blank" rel="noopener noreferrer" style={{textDecoration: 'none', color: 'inherit'}}>Gnosis Research Center (GRC)</a>
              <img src="/img/logos/iit-logo.png" alt="IIT" className="hero__eyebrowLogoLarge" />
            </div>

            {/* Logo + Brand name horizontal */}
            <div className="hero__branding">
              <img
                src="/img/iowarp_logo.png"
                alt="IoWarp Logo"
                className="hero__logo"
              />
              <div className="hero__brandGroup">
                <h1 className="hero__brand">CLIO KIT</h1>
                <p className="hero__brandSub">Part of the <a href="https://iowarp.ai" target="_blank" rel="noopener noreferrer">IoWarp Platform</a></p>
              </div>
            </div>

            {/* Title */}
            <h2 className="hero__title">
              Talk to Data, Devices, Apps<br/>
              A Meta-Marketplace for Agents
            </h2>

            {/* Buttons */}
            <div className="hero__actions">
              <Link className="button button--primary hero__cta" to="/docs/intro">
                Install in Seconds
              </Link>
              <a className="button button--outline hero__cta" href="#browse">
                Explore Servers
              </a>
              <Link className="button button--ghost hero__cta" href="https://github.com/iowarp/clio-kit" rel="noopener noreferrer">
                Star on GitHub
              </Link>
            </div>

            {/* Subtitle - 2 lines */}
            <p className="hero__subtitle">
              A meta-marketplace for scientific AI agents.<br/>
              22 MCP servers, skills, plugins, agents, and community contributions.<br/>
              Works with <a href="https://www.claude.com/product/claude-code" target="_blank" rel="noopener">Claude Code</a>,{' '}
              <a href="https://cursor.com/home" target="_blank" rel="noopener">Cursor</a>,{' '}
              <a href="https://code.visualstudio.com/" target="_blank" rel="noopener">VS Code</a>,{' '}
              <a href="https://github.com/openai/codex" target="_blank" rel="noopener">Codex CLI</a>,{' '}
              <a href="https://antigravity.google/" target="_blank" rel="noopener">Antigravity</a>,{' '}
              <a href="https://github.com/sst/opencode" target="_blank" rel="noopener">OpenCode</a> and other clients.
              {' '}<Link to="/docs/intro#agent-integrations">MCP, skill, and plugin setup</Link>.
            </p>

            {/* Marketplace features */}
            <div className="hero__highlights">
              <div className="hero__highlightCard">
                <h3>Scientific MCP Servers</h3>
                <p>
                  Explore data, analyze results, search papers, and manage HPC jobs
                  with 22 MCP servers and hybrid search.{' '}
                  <a href="#browse">Browse servers</a>.
                </p>
              </div>

              <div className="hero__highlightCard">
                <h3>Skills, Bundles &amp; Agents</h3>
                <p>
                  Use 20 skills and six workflow bundles, with two optional planning
                  and review agents for Claude Code.{' '}
                  <Link to="/docs/intro#agent-integrations">Choose your agent setup</Link>.
                </p>
              </div>

              <div className="hero__highlightCard">
                <h3>Community Contributions</h3>
                <p>
                  Share skills, plugins, or MCP servers in any language. Index
                  external marketplace collections while maintainers keep control
                  of their code and releases.{' '}
                  <Link to="/docs/marketplace#contributing-and-updating">Contribute</Link>.
                </p>
              </div>
            </div>

            {/* Footer - Single line */}
            <div className="hero__footer">
              Part of the <a href="https://iowarp.ai" className="hero__footerLink" target="_blank" rel="noopener noreferrer"><strong>IoWarp Platform</strong></a> · Open-Source Community Project supported in part by the{' '}
              <img src="/img/logos/nsf-logo.png" alt="NSF" className="hero__nsfLogo" />
              <a href="https://new.nsf.gov/" className="hero__footerLink" target="_blank" rel="noopener noreferrer">
                National Science Foundation (NSF)
              </a>
            </div>
          </div>
        </section>

        {/* MCP Showcase */}
        <div id="browse">
          <MCPShowcase />
        </div>

      </main>
    </Layout>
  );
}
