# Traditional Research Workflows in Material Science

## 1. Experimental Research Workflow
**Best for**: New material synthesis, property measurement, structure-property relationships

```
Task 01: Literature review & gap identification
Task 02: Hypothesis formulation
Task 03: Experimental design (DOE, sample matrix)
Task 04: Sample/specimen preparation
Task 05: Characterization (structure)
Task 06: Property measurement (mechanical / functional / etc.)
Task 07: Data analysis & statistical treatment
Task 08: Interpretation & mechanism identification
Task 09: Write-up & publication
```

**Key assumptions to interview about each task**:
- Task 03: What variables are being controlled? What is the sample size?
- Task 04: What processing routes are available? What are contamination risks?
- Task 05: What characterization tools are accessible? What resolution is needed?
- Task 06: What standards (ASTM, ISO) apply? What test conditions?
- Task 07: What statistical methods? Are outliers expected?

---

## 2. Computational Research Workflow
**Best for**: Property prediction, screening, mechanism understanding, new hypothesis generation

```
Task 01: Literature review & gap identification
Task 02: Problem formulation & model selection
Task 03: Parameter validation (benchmarking vs. experiment)
Task 04: Main simulations / calculations
Task 05: Convergence & sensitivity testing
Task 06: Data analysis & visualization
Task 07: Interpretation & physical insight
Task 08: Experimental validation (if applicable)
Task 09: Write-up & publication
```

**Key assumptions per task**:
- Task 02: DFT vs. MD vs. phase-field? What exchange-correlation functional?
- Task 03: What experimental data exists for benchmarking?
- Task 04: What system sizes and timescales are computationally feasible?
- Task 05: What convergence criteria are acceptable?

---

## 3. Integrated Experimental + Computational Workflow
**Best for**: Mechanistic studies, structure-property-processing relationships

```
Task 01: Literature review & gap identification
Task 02: Hypothesis formulation
Task 03: Computational screening / pre-selection
Task 04: Experimental design based on computational insights
Task 05: Sample preparation
Task 06: Characterization
Task 07: Property measurement
Task 08: Computational validation / refinement
Task 09: Mechanism elucidation (combined)
Task 10: Write-up & publication
```

---

## 4. Materials Optimization Workflow
**Best for**: Alloy design, processing parameter optimization, performance maximization

```
Task 01: Literature review; identify composition/process space
Task 02: Design of experiments (DOE) or high-throughput plan
Task 03: Sample library preparation
Task 04: Rapid characterization (screening)
Task 05: Property measurement (screening)
Task 06: Down-selection & focused study
Task 07: Detailed characterization of top candidates
Task 08: Mechanism analysis
Task 09: Performance validation (application-relevant conditions)
Task 10: Write-up & publication
```

---

## 5. Literature Review / Meta-Analysis Workflow
**Best for**: Review papers, meta-analysis, identifying trends across datasets

```
Task 01: Define scope & inclusion/exclusion criteria
Task 02: Database search & paper collection
Task 03: Abstract screening
Task 04: Full-text review & data extraction
Task 05: Data harmonization (units, conditions, definitions)
Task 06: Statistical analysis / meta-analysis
Task 07: Trend identification & gap analysis
Task 08: Synthesis & narrative development
Task 09: Write review paper
```

---

## 6. Device / Application Workflow
**Best for**: Proof-of-concept demonstration, prototype development

```
Task 01: Literature review; material requirements & state of art
Task 02: Material selection / design
Task 03: Fabrication process development
Task 04: Device assembly
Task 05: Performance testing (device-level)
Task 06: Degradation / lifetime study
Task 07: Benchmarking vs. state-of-art
Task 08: Write-up & publication
```

---

## Task Type Definitions

| Type | What It Means | Agent Behavior |
|------|---------------|----------------|
| `literature` | Search, read, synthesize papers | Review supplied corpus and permitted provided URLs, then summarize → update LITERATURE.md |
| `experimental` | Physical lab work | Generate protocol + recording template + checklist |
| `computational` | Simulation / calculation | Generate input files + analysis scripts |
| `analytical` | Mathematical modeling, theory | Derive equations + validate against data |
| `data-analysis` | Process existing data | Write analysis code + produce figures |
| `writing` | Draft paper sections | Prepare author-reviewed paper handoff |

---

## Checkpoint Placement Guidelines

Place `checkpoint:human-action` when:
- Experimental data needs to be uploaded before analysis can proceed
- A go/no-go decision is needed based on preliminary results
- Equipment access or resource availability must be confirmed

Place `checkpoint:decision` when:
- Two valid experimental paths exist (e.g., XRD vs. neutron diffraction)
- Computational model must be chosen (DFT vs. MD vs. phase-field)
- Scope must be narrowed after unexpected preliminary results

Place `checkpoint:human-verify` when:
- Assumptions have been generated and need domain expert sign-off
- Analysis results need interpretation before the next task begins
