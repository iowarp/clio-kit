# Virtual Lab

## Experimental Facilities

### Labs Accessible
| Lab Type | Location/Name | Access Level | Notes |
|----------|--------------|--------------|-------|
| [Chemistry/wet lab] | [building/room] | [daily | scheduled | limited] | [] |
| [Mechanical testing] | [building/room] | [daily | scheduled | limited] | [] |
| [Microscopy suite] | [building/room] | [daily | scheduled | limited] | [] |
| [Cleanroom] | [building/room] | [daily | scheduled | limited] | [] |
| [Processing/synthesis] | [building/room] | [daily | scheduled | limited] | [] |

### Equipment Inventory

#### Characterization
| Equipment | Model/Capability | Access | Booking | Limitations |
|-----------|-----------------|--------|---------|-------------|
| SEM | [model, kV range, EDS?] | [in-house|external] | [yes/no, lead time] | [sample size, conductivity] |
| TEM/STEM | [model, EELS/EDS?] | [in-house|external] | [yes/no, lead time] | [sample prep required] |
| XRD | [model, Cu/Mo source, SAXS?] | [in-house|external] | [yes/no] | [powder/bulk/thin film] |
| XPS | [model] | [in-house|external] | [yes/no] | [] |
| APT | [model] | [in-house|external] | [yes/no] | [] |
| Raman | [model] | [in-house|external] | [yes/no] | [] |
| DSC/TGA | [model, temp range] | [in-house|external] | [yes/no] | [] |

#### Mechanical Testing
| Equipment | Capability | Access | Max Load | Temperature Range |
|-----------|-----------|--------|----------|-------------------|
| Universal testing machine | [tensile/compression/fatigue] | [in-house|external] | [kN] | [°C] |
| Fatigue tester | [load-controlled, displacement] | [in-house|external] | [kN] | [] |
| Hardness tester | [Vickers/Rockwell/nano] | [in-house|external] | [] | [] |
| Nanoindenter | [model] | [in-house|external] | [mN] | [] |
| DMA | [model, freq range] | [in-house|external] | [] | [] |

#### Processing / Synthesis
| Equipment | Type | Capability | Access |
|-----------|------|-----------|--------|
| Furnace | [box/tube/vacuum] | [max temp °C, atmosphere] | [in-house|external] |
| Arc melter | [] | [button size, inert gas] | [in-house|external] |
| Electrospinning | [] | [] | [in-house|external] |
| Sputtering/PVD | [] | [targets, substrate size] | [in-house|external] |
| 3D printer / SLM | [material, build volume] | [] | [in-house|external] |

---

## Computational Resources

### HPC Systems
| System Name | Cores Available | RAM/Node | Storage | Scheduler | Key Software |
|-------------|----------------|---------|---------|-----------|-------------|
| [cluster name] | [N cores/job] | [GB] | [TB quota] | [SLURM|PBS|SGE] | [VASP, LAMMPS, Abaqus, etc.] |

### Local Workstations
| Machine | CPU | GPU | RAM | OS | Special Use |
|---------|-----|-----|-----|----|-------------|
| [desktop/laptop] | [cores, GHz] | [model, VRAM] | [GB] | [Linux/Win] | [primary workstation] |

### Cloud / External Compute
| Resource | Type | Cost Model | Access Method |
|----------|------|-----------|---------------|
| [AWS/GCP/Azure/XSEDE/ACCESS] | [CPU/GPU cluster] | [$/hr or allocation] | [always|limited] |

---

## Software Licenses
| Software | Purpose | License | Version | Notes |
|----------|---------|---------|---------|-------|
| VASP | DFT | [group|institutional] | [6.x] | [] |
| LAMMPS | MD | open-source | [] | [] |
| Thermo-Calc / Pandat | CALPHAD | [group|institutional] | [] | [] |
| Abaqus | FEM | [group|institutional] | [] | [] |
| VESTA | Crystal viz | free | [] | [] |
| Python (numpy, scipy, pymatgen, ASE) | Analysis | open-source | [] | [] |
| MATLAB | Analysis | [institutional] | [] | [] |
| OriginPro | Plotting | [group|institutional] | [] | [] |

---

## External Facilities & Collaborations
| Facility | Type | Access Route | Typical Lead Time | Contact |
|----------|------|-------------|-------------------|---------|
| [National lab, e.g., APS, NSLS-II] | [synchrotron XRD] | [proposal|collaborator] | [months] | [] |
| [University core facility] | [] | [fee-for-service] | [weeks] | [] |
| [Industry collaborator] | [] | [MOU|collaboration] | [] | [] |

---

## Materials & Consumables
- **Budget**: [$/month or total project budget]
- **Material sourcing**: [in-house synthesis | purchased (supplier) | collaborator-provided]
- **Typical lead time**: [days/weeks for material procurement]
- **Storage**: [controlled atmosphere | cryogenic | ambient]

---

## Human Resources
- **Personnel**: [PI only | + grad students (N) | + postdoc | + technician | + undergrads]
- **Time on this project**: [hours/week]
- **Key skills in lab**: [list relevant expertise]
- **Skills gaps**: [techniques nobody in the lab knows yet]

---

## Constraints & Blockers
- [Equipment currently down or unavailable]
- [Booking backlogs or seasonal access issues]
- [Safety certifications required but not yet held]
- [Export control or IP restrictions on certain materials/software]

---

## Resource-to-Task Mapping
*Auto-generated during define-virtual-lab; maps available resources to WORKFLOW.md task types*

| Task Type | Required Resource | Available? | Alternative if Not |
|-----------|-----------------|------------|-------------------|
| XRD characterization | In-house XRD | [yes/no] | [external facility X] |
| DFT calculation | HPC with VASP | [yes/no] | [cloud compute, QE open-source] |
| Tensile testing | UTM | [yes/no] | [external lab Y] |
