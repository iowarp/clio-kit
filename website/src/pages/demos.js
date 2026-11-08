import React, {useState} from 'react';
import Link from '@docusaurus/Link';
import {Frame} from '../components/Marketplace/shared';
import s from '../components/Marketplace/overview.module.css';
import d from './demos.module.css';

// Recordings from before CLIO Kit 1.0, hosted on the project's YouTube channel.
const demos = [
  {
    id: 'paraview',
    server: 'ParaView MCP',
    title: 'Render a pipe-flow simulation',
    emphasis: 'in a live ParaView session.',
    summary:
      'The agent loads an InCompact3d BP5 dataset into ParaView, adds pressure and velocity isosurfaces and three orthogonal slices, and saves a screenshot after each step.',
    facts: [
      ['Version', 'Older CLIO Kit, then called Agent Toolkit'],
      ['Recorded', 'November 2025'],
      ['Agent', 'Claude Code · Sonnet 4.5'],
      ['Data', 'InCompact3d pipe flow, BP5 · ParaView 5.13.1'],
    ],
    videos: [
      {
        youtube: 'okcmEgnzJEw',
        title: 'Claude Code and ParaView side by side',
        poster: '/img/demos/paraview-action.webp',
        ratio: 2234 / 1374,
        length: '3:08',
      },
      {
        youtube: 'a3ugZcTiU2s',
        title: 'The terminal session',
        poster: '/img/demos/paraview-session.webp',
        ratio: 1,
        length: '3:00',
      },
    ],
    prompts: [
      'Load and analyze the BP5 dataset from @bp5-dataset-collection/InCompact3d/Pipe-Flow/data.bp5 and perform comprehensive fluid dynamics visualization. Context: This is a computational fluid dynamics dataset containing pressure and velocity field data from a pipe flow simulation.',
      'Load the BP5 dataset and examine available fields and data structure.',
      'Create an isosurface visualization for pressure (pp) field at value 0.1 with appropriate color mapping.',
      'Create three orthogonal slice planes for comprehensive flow analysis: Y-normal slice (horizontal plane) at origin, Z-normal slice (vertical plane) at origin, X-normal slice (front view) at origin.',
      'Create an isosurface for velocity magnitude using ux field at value 0.5 with distinct coloring.',
    ],
    tools: [
      'load_scientific_data',
      'generate_isosurface',
      'create_data_slice',
      'set_representation_type',
      'apply_field_coloring',
      'set_active_source',
    ],
    about: {
      description:
        'Showcase of IOWarp’s ParaView MCP integration for comprehensive computational fluid dynamics (CFD) visualization, demonstrating automated analysis and rendering of BP5 datasets from pipe flow simulations.',
      points: [
        [
          'CFD Data Analysis',
          'Processing BP5 files from InCompact3d pipe flow simulations containing pressure and velocity fields',
        ],
        [
          'Automated Visualization Pipeline',
          'Creating sophisticated visualizations through natural language commands without manual ParaView GUI interaction',
        ],
        [
          'Isosurface Generation',
          'Rendering pressure and velocity magnitude isosurfaces to identify critical flow features',
        ],
        [
          'Multi-Plane Slicing',
          'Generating orthogonal slice planes (X, Y, Z) for comprehensive three-dimensional flow field analysis',
        ],
        [
          'Interactive Rendering',
          'Producing high-quality visualizations with appropriate color mapping for scientific interpretation',
        ],
      ],
      summary:
        'The demo showcases how IOWarp’s ParaView MCP integration enables researchers to perform complex 3D visualization tasks using simple natural language instructions, making advanced CFD analysis more accessible and streamlining the workflow from data to publication-ready visualizations.',
    },
    outputs: [
      [
        'paraview-isosurface-pp',
        'isosurface_pipeflow_visual.png',
        'The pressure isosurface at pp = 0.1.',
        1600,
        879,
      ],
      [
        'paraview-contour',
        'contour.png',
        'The Contour1 filter rendered as a surface.',
        1600,
        879,
      ],
      [
        'paraview-slice1',
        'slice1.png',
        'The Y-normal slice, Slice1.',
        1600,
        879,
      ],
      [
        'paraview-slice2',
        'slice2.png',
        'The Z-normal slice, Slice2.',
        1600,
        879,
      ],
      [
        'paraview-slice3',
        'slice3.png',
        'The X-normal slice, Slice3.',
        1600,
        879,
      ],
    ],
    files: [
      [
        '/img/demos/paraview-views.webp',
        'paraview_rander_output.png',
        'Six of the saved views on one labeled sheet.',
      ],
    ],
    current: ['mcp/paraview', 'ParaView'],
  },
  {
    id: 'adios',
    server: 'ADIOS2 MCP',
    title: 'Question a molecular dynamics run',
    emphasis: 'in plain language.',
    summary:
      'The agent reads a LAMMPS gold-melting simulation through the ADIOS2 server, answers questions about the system, and plots the final atom positions and two single-atom trajectories.',
    facts: [
      ['Version', 'Older CLIO Kit, then called IOWarp MCPs'],
      ['Recorded', 'September 2025'],
      ['Agent', 'Claude Code'],
      ['Data', 'LAMMPS gold-melting run, BP5'],
    ],
    videos: [
      {
        youtube: 'xf-Ysp_gPZ8',
        title: 'The terminal session',
        poster: '/img/demos/adios.webp',
        ratio: 1,
        length: '9:47',
      },
    ],
    prompts: [
      'What can you tell me about the simulation based on the dataset.',
      'What was the initial temperature of the system?',
      'How many gold atoms are in the simulation?',
      'What are the dimensions of the simulation box?',
      'What is the total duration of the simulation in picoseconds?',
      'What is the crystal structure of the gold at the beginning of the simulation?',
      'Plot the positions of the atoms at the final timestep?',
      'Plot the trajectory of a single atom over time. The atom of choice should a parameter to the script. The output of the script should a PNG image with the results. Run the script for any single atom',
    ],
    tools: [
      'list_bp5',
      'inspect_variables',
      'inspect_attributes',
      'read_variable_at_step',
    ],
    about: {
      description:
        'Showcase of IOWarp ADIOS MCP providing full analysis of a BP5 file generated by LAMMPS, demonstrating natural language queries on scientific data.',
      points: [
        [
          'Scientific Data Interpretation',
          'Analyzing BP5 files generated by LAMMPS molecular dynamics simulations',
        ],
        [
          'Natural Language Queries',
          'Enabling researchers to ask complex questions about simulation data in plain English',
        ],
        [
          'Automated Analysis',
          'Extracting key simulation parameters like temperature, atom count, box dimensions, and duration',
        ],
        [
          'Visualization Generation',
          'Creating plots and visualizations of atomic positions and trajectories',
        ],
        [
          'Interactive Exploration',
          'Providing an intuitive interface for exploring complex scientific datasets',
        ],
      ],
      summary:
        'The demo showcases how IOWarp’s ADIOS MCP can bridge the gap between complex scientific data formats and accessible analysis tools, making it easier for researchers to understand and visualize molecular dynamics simulations of gold melting processes.',
    },
    outputs: [
      [
        'adios-final-positions',
        'final_positions.png',
        'Atom positions at the final timestep, 26,000.',
        1600,
        1445,
      ],
      [
        'adios-final-positions-xy',
        'final_positions_xy.png',
        'The same positions projected onto x–y, colored by z.',
        1600,
        1387,
      ],
      [
        'adios-atom-1000',
        'atom_1000_trajectory.png',
        'Trajectory of atom 1000.',
        1600,
        1195,
      ],
      [
        'adios-atom-5000',
        'atom_5000_trajectory.png',
        'Trajectory of atom 5000.',
        1600,
        1195,
      ],
    ],
    current: ['mcp/adios', 'ADIOS2'],
  },
  {
    id: 'ndp',
    server: 'NDP, Pandas and Plot MCPs',
    title: 'Find a dataset, explore it',
    emphasis: 'and plot what it contains.',
    summary:
      'The agent finds the latest EarthScope dataset through the National Data Platform, downloads its GeoJSON metadata and GNSS time series, profiles the table, and writes a combined three-axis figure.',
    facts: [
      ['Version', 'Older CLIO Kit, then called IOWarp MCPs'],
      ['Recorded', 'October 2025'],
      ['Agent', 'Claude Code'],
      ['Data', 'EarthScope GNSS station ODSA, 813,013 rows'],
    ],
    videos: [
      {
        youtube: 'poJ-8gg5pzk',
        title: 'The terminal session',
        poster: '/img/demos/ndp.webp',
        ratio: 1,
        length: '1:13',
      },
    ],
    prompts: [
      'Use the ndp-mcp to find the latest dataset of the earthscope organization, find the url of the geonjson and csv they contain, and curl them.',
      'The geojson is metadata, the csv contains seismograph data.',
      'Use the pandas-mcp to explore the csv file and understand the data.',
      'Use the plot-mcp to plot a line plot of each of the axis.',
      'Using uv to manage it, create a final plot that combines all of the axis as subgraphs into a single graph including as much information as possible from the gathered metadata.',
    ],
    tools: [
      'ndp-mcp list_organizations',
      'pandas-mcp profile_data',
      'plot-mcp line_plot',
    ],
    about: {
      description:
        'Showcase of NDP MCP providing full analysis of EarthScope seismograph data, demonstrating natural language queries on geophysical datasets with comprehensive visualization.',
    },
    outputs: [
      [
        'ndp-gnss-analysis',
        'comprehensive_gnss_analysis.png',
        'East, north and vertical displacement with station metadata.',
        1600,
        1398,
      ],
    ],
    files: [
      [
        '/img/demos/comprehensive_seismic_plot.py',
        'comprehensive_seismic_plot.py',
        'The plotting script the agent wrote and ran with uv.',
      ],
    ],
    current: ['mcp/ndp', 'NDP'],
  },
  {
    id: 'agentlog',
    server: 'ChronoLog MCP',
    title: 'Keep a record of agent sessions',
    emphasis: 'and recall it later.',
    summary:
      'AgentLog writes the agent’s session events to ChronoLog. Earlier sessions ask about the machine and about NOAA datasets; a later session retrieves those answers from the log.',
    facts: [
      ['Version', 'Older CLIO Kit, then called IOWarp MCPs'],
      ['Recorded', 'September 2025'],
      ['Agent', 'Claude Code'],
      ['Data', 'ChronoLog event log, National Data Platform'],
    ],
    videos: [
      {
        youtube: 'uIfAwh3g0SU',
        title: 'Three sessions in sequence',
        poster: '/img/demos/agent-log.webp',
        ratio: 1,
        length: '1:34',
      },
    ],
    prompts: [
      'show my cpu information',
      'show network information',
      'List all organizations in the National Data Platform to see what data is available',
      'Find the climate datasets url from NOAA',
      'What was the NOAA URLs: NDFD we found in a previous conversation (use chronolog-mcp)',
      'What was the machine information we found in previous conversation (use chronolog-mcp)',
    ],
    tools: [
      'chronolog-mcp',
      'ndp-mcp list_organizations',
      'ndp-mcp search_datasets',
      'ndp-mcp get_dataset_details',
    ],
    about: {
      description:
        'Showcase of AgentLog (ChronoLog MCP) providing full analysis and observability of interactive session events, demonstrating natural language queries on distributed logging data with comprehensive event tracking.',
    },
    recalled: [
      'NDFD HTTPS access: https://www.ncei.noaa.gov/data/national-digital-forecast-database/access/',
      'NDFD THREDDS server: https://www.ncei.noaa.gov/thredds/model/ndfd.html',
      'Intel Xeon Silver 4114 · 20 physical and 40 logical cores · 4.68% average CPU use',
    ],
    current: ['mcp/chronolog', 'ChronoLog'],
  },
];

function Video({youtube, title, poster, ratio, length}) {
  // ponytail: a still frame until the visitor presses play; YouTube loads only then.
  const [playing, setPlaying] = useState(false);
  return (
    <figure className={d.video}>
      <div className={d.screen} style={{aspectRatio: ratio}}>
        {playing ? (
          <iframe
            src={`https://www.youtube-nocookie.com/embed/${youtube}?autoplay=1&rel=0`}
            title={title}
            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
            referrerPolicy="strict-origin-when-cross-origin"
            allowFullScreen
          />
        ) : (
          <button
            type="button"
            onClick={() => setPlaying(true)}
            aria-label={`Play: ${title} (${length}), recorded with an older version of CLIO Kit`}
          >
            <img src={poster} alt="" loading="lazy" />
            <span className={d.version} aria-hidden="true">
              Older version of CLIO Kit
            </span>
            <span className={d.play} aria-hidden="true">
              ▶ Play · {length}
            </span>
          </button>
        )}
      </div>
      <figcaption>
        {title}{' '}
        <a href={`https://www.youtube.com/watch?v=${youtube}`}>
          YouTube <span aria-hidden="true">↗</span>
        </a>
      </figcaption>
    </figure>
  );
}

function Demo({demo, number}) {
  const columns = demo.videos.map((video) => `${video.ratio}fr`).join(' ');
  return (
    <section
      id={demo.id}
      className={d.demo}
      aria-labelledby={`${demo.id}-title`}
    >
      <div className={d.demoHeading}>
        <div>
          <p className={s.eyebrow}>
            <span>{String(number).padStart(2, '0')} /</span> {demo.server}
          </p>
          <h2 id={`${demo.id}-title`}>
            {demo.title}
            <br />
            <em>{demo.emphasis}</em>
          </h2>
          <p>{demo.summary}</p>
        </div>
        <dl className={d.facts}>
          {demo.facts.map(([term, value]) => (
            <div key={term}>
              <dt>{term}</dt>
              <dd>{value}</dd>
            </div>
          ))}
        </dl>
      </div>
      {/* One video sits beside the details; a pair spans the page above them. */}
      <div
        className={d.body}
        data-layout={demo.videos.length === 1 ? 'aside' : 'stack'}
      >
        <div className={d.videos} style={{'--columns': columns}}>
          {demo.videos.map((video) => (
            <Video key={video.youtube} {...video} />
          ))}
        </div>
        <div className={d.details}>
          <div>
            <h3>What was asked</h3>
            <ol className={d.prompts}>
              {demo.prompts.map((prompt) => (
                <li key={prompt}>{prompt}</li>
              ))}
            </ol>
          </div>
          <div>
            <h3>MCP calls in the recording</h3>
            <ul className={d.tools}>
              {demo.tools.map((tool) => (
                <li key={tool}>
                  <code>{tool}</code>
                </li>
              ))}
            </ul>
            <h3>What it demonstrates</h3>
            <p>{demo.about.description}</p>
            {demo.about.points && (
              <ol className={d.points}>
                {demo.about.points.map(([label, text]) => (
                  <li key={label}>
                    <strong>{label}:</strong> {text}
                  </li>
                ))}
              </ol>
            )}
            {demo.about.summary && <p>{demo.about.summary}</p>}
            <Link className={s.textLink} to={`/catalogue/${demo.current[0]}`}>
              Use the current {demo.current[1]} server <span>→</span>
            </Link>
          </div>
        </div>
      </div>
      <div className={d.produced}>
        <h3>What it produced</h3>
        {demo.outputs && (
          <div
            className={d.outputs}
            style={{
              '--output-columns':
                demo.outputs.length === 4
                  ? 2
                  : Math.min(demo.outputs.length, 3),
              maxWidth: demo.outputs.length === 1 ? '56rem' : undefined,
            }}
          >
            {demo.outputs.map(([image, file, caption, width, height]) => (
              <figure key={image}>
                <a href={`/img/demos/${image}.webp`}>
                  <img
                    src={`/img/demos/${image}.webp`}
                    width={width}
                    height={height}
                    loading="lazy"
                    alt={caption}
                  />
                </a>
                <figcaption>
                  <code>{file}</code> {caption}
                </figcaption>
              </figure>
            ))}
          </div>
        )}
        {demo.files?.map(([href, file, caption]) => (
          <p key={file} className={d.file}>
            <a href={href} download={href.endsWith('.py')}>
              <code>{file}</code>
            </a>{' '}
            {caption}
          </p>
        ))}
        {demo.recalled && (
          <div className={d.recalled}>
            <p className={s.eyebrow}>
              Recalled from ChronoLog in a new session
            </p>
            <ul>
              {demo.recalled.map((line) => (
                <li key={line}>{line}</li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </section>
  );
}

export default function Demos() {
  return (
    <Frame
      title="Demos"
      description="Demos recorded with an older version of CLIO Kit: ParaView, ADIOS2, NDP and ChronoLog servers in Claude Code, with the prompts and every output file."
    >
      <div className={`${s.home} ${s.page}`}>
        <header className={s.tutorialIntro}>
          <p className={s.eyebrow}>CLIO Kit / Demos</p>
          <h1>Watch an agent work with scientific data.</h1>
          <p className={s.lede}>
            These demos use an older version of CLIO Kit. Each one shows the
            requests that started the session and the files it produced.
          </p>
          <div className={d.notice}>
            <span className={s.tag}>Older version</span> Recorded in 2025 with
            an older version of CLIO Kit, before 1.0, when the project was
            called IOWarp MCPs and then Agent Toolkit. Server names, tools and
            install commands have changed since. For the current version,{' '}
            <Link to="/tutorials">follow the tutorials</Link>.
          </div>
          <nav className={s.chapterNav} aria-label="Demos">
            {demos.map((demo) => (
              <a key={demo.id} href={`#${demo.id}`}>
                {demo.server}
              </a>
            ))}
          </nav>
        </header>
        {demos.map((demo, index) => (
          <Demo key={demo.id} demo={demo} number={index + 1} />
        ))}
      </div>
    </Frame>
  );
}
