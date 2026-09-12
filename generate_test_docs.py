import pymupdf
from pathlib import Path

docs_dir = Path("./documents")
docs_dir.mkdir(parents=True, exist_ok=True)

# 10 Multi-Page Document Definitions (Minimum 5 pages each)
DOCS_DATA = [
    {
        "filename": "ev_battery_standard_r100_rev2021.pdf",
        "title": "ECE R100 Rev 2021: Specific Requirements for Electric Powertrain Safety",
        "doc_id": "STD-R100-2021",
        "year": 2021,
        "pages": [
            [
                "ECE Regulation No. 100 Revision 2021: Uniform Provisions Concerning Vehicle Safety.",
                "1. Scope and Field of Application. This standard applies to the electric power train of road vehicles of categories M and N.",
                "Section 1.1: High Voltage Safety Protocols. Electrical components must maintain a nominal working voltage exceeding 60 V DC and up to 1500 V DC.",
                "All live high voltage parts must possess IPXXB touch protection under normal operating conditions. Isolation resistance must be at least 100 ohms per volt for DC circuits."
            ],
            [
                "2. Battery Crash Protection and Mechanical Shock Standards (2021 Release).",
                "Paragraph 2.1: Mechanical impact deceleration testing requires a minimum pulse duration of 65 milliseconds with peak acceleration reaching 28g.",
                "Paragraph 2.2: Structural integrity verification dictates that no electrolyte leakage shall occur within 30 minutes following dynamic deceleration.",
                "Deformation of the outer enclosure must not penetrate internal cell casing boundaries under 100 kN static compressive load."
            ],
            [
                "3. Thermal Runaway Propagation Standards (2021 Baseline Thresholds).",
                "Paragraph 3.1: Initiation criteria. A single cell thermal runaway is triggered via a 200W ceramic heating cartridge.",
                "Paragraph 3.2: Mitigation time window. The Battery Management System (BMS) must provide an audible alarm at least 5 minutes prior to hazardous conditions.",
                "The 2021 standard does not mandate complete suppression of fire outside the pack enclosure after 10 minutes of propagation."
            ],
            [
                "4. Chemical Composition and Heavy Metal Limitations (2021 Guidelines).",
                "Section 4.1: Cobalt utilization threshold. Cathode formulations must not exceed 20% cobalt weight ratio in automotive commercial traction cells.",
                "Section 4.2: Recycled materials quota. In 2021, recycled lithium recovery targets remain voluntary with an advisory target of 25% recovery efficiency.",
                "Nickel-Manganese-Cobalt (NMC 622) is established as the certified reference chemistry for compliance bench tests."
            ],
            [
                "5. Verification Annex and Sign-off Criteria (2021 Cycle).",
                "Annex A: State of Health (SoH) degradation thresholds allow a 25% loss over 8 years or 160,000 kilometers.",
                "Annex B: Type approval certification requires testing across 5 production-line battery packs under ambient conditions of 20°C ± 5°C.",
                "Authority sign-off issued under Standard ECE-R100-2021. All revisions supersede 2018 guidelines."
            ]
        ]
    },
    {
        "filename": "ev_battery_standard_r100_rev2023.pdf",
        "title": "ECE R100 Rev 2023: Advanced Thermal Safety and Fast Charging Revisions",
        "doc_id": "STD-R100-2023",
        "year": 2023,
        "pages": [
            [
                "ECE Regulation No. 100 Revision 2023: Addendum on Ultra-Fast Charging and High Current Safety.",
                "1. Scope and Revisions. Supersedes the 2021 baseline with stringent protocols for high-rate charging infrastructures above 250 kW.",
                "Section 1.1: DC Isolation Monitoring. Insulation resistance requirement is raised from 100 ohms/volt to 500 ohms/volt under humid operating conditions.",
                "BMS ground fault detection circuits must trigger high-voltage disconnect contactors within 100 milliseconds of threshold breach."
            ],
            [
                "2. Dynamic Mechanical Crush and Penetration Updates (2023 Standards).",
                "Paragraph 2.1: Nail penetration testing is reinstated for non-LFP chemistries using a 5mm tungsten carbide pin moving at 20 mm/sec.",
                "Paragraph 2.2: Enclosure ingress rating is formally upgraded to IP6K9K to ensure resistance against high-pressure steam jet cleaning.",
                "Electrolyte leakage after static 100 kN crush test is strictly prohibited; zero milliliter discharge is the absolute pass criterion."
            ],
            [
                "3. Extended Thermal Runaway Protection (2023 Enforcement).",
                "Paragraph 3.1: The driver early warning window is extended from 5 minutes to 15 minutes prior to smoke or gas ingress into the passenger cabin.",
                "Paragraph 3.2: Venting gas flammability. Pack enclosures must incorporate directional exhaust burst discs designed to vent away from passenger zones.",
                "Internal cell-to-cell thermal barrier aerogel blankets must sustain 800°C for a minimum duration of 20 minutes."
            ],
            [
                "4. Heavy Metal Ceilings and Material Tracking (2023 Revision).",
                "Section 4.1: Cobalt reduction limit. Cobalt weight ratio in NMC cathodes is strictly capped at 10% (transitioning toward NMC 811 formulations).",
                "Section 4.2: Closed-loop recycling quotas. Manufacturers must demonstrate that at least 50% of active lithium and 70% of cobalt are recovered during end-of-life.",
                "Battery passport digital traceability is introduced as a voluntary audit provision."
            ],
            [
                "5. Warranty and Capacity Retention Standards (2023 Requirements).",
                "Annex A: Capacity retention mandate is tightened: maximum allowable degradation is 20% over 8 years or 160,000 kilometers.",
                "Annex B: High-rate fast charging test mandates 500 consecutive cycles at 3C rate with less than 8% non-linear degradation.",
                "Compliance sign-off registered under ECE-R100-2023, superseding the 2021 protocol."
            ]
        ]
    },
    {
        "filename": "ev_battery_standard_r100_rev2026.pdf",
        "title": "ECE R100 Rev 2026: Solid-State Integration and Zero-Propagation Mandate",
        "doc_id": "STD-R100-2026",
        "year": 2026,
        "pages": [
            [
                "ECE Regulation No. 100 Revision 2026: Solid-State and Advanced Cell Architecture Framework.",
                "1. Scope: Formal guidelines encompassing both liquid electrolyte lithium-ion and solid-state electrolyte battery architectures.",
                "Section 1.1: Operating Voltage Ceiling. Standardized protocols updated for 1200 V DC high-voltage architectures.",
                "Isolation monitoring systems must execute self-diagnostic tests continuously every 10 milliseconds, maintaining 1000 ohms per volt."
            ],
            [
                "2. Zero-Propagation Thermal Safety Mandate (2026 Definitive Rule).",
                "Paragraph 2.1: The 15-minute warning window from 2023 is replaced by a complete 'Zero-Propagation' standard.",
                "Under thermal runaway of cell #1, no propagation to neighboring cells is permissible under any circumstances.",
                "Passenger compartment warning time is increased to 30 minutes, during which external pack surface temperatures must not exceed 60°C."
            ],
            [
                "3. Advanced Mechanical Integrity and Submersion Standards (2026 Revision).",
                "Paragraph 3.1: Saltwater submersion test. The entire energized pack is immersed in 3.5% NaCl solution for 2 hours at depth of 1 meter.",
                "No explosion, fire, or arc discharge is permitted during submersion or the subsequent 24-hour observation window.",
                "Mechanical crush criteria require pack structural withstand capability of 150 kN omnidirectional force."
            ],
            [
                "4. Cathode Sustainability and Battery Passport (2026 Mandatory Standards).",
                "Section 4.1: Cobalt ceiling. Cobalt content must not exceed 5% by total active cathode mass. Cobalt-free (LFP, LMFP, Na-ion) formats incentivized.",
                "Section 4.2: Mandatory Battery Passport. Every commercial pack sold in 2026 must encode an immutable cryptographic passport detailing carbon footprint.",
                "Recycling mandate requires 80% lithium recovery and 95% nickel/cobalt recovery efficiency verified by third-party audit."
            ],
            [
                "5. Lifecycle Guarantees and Certification Protocols (2026 Release).",
                "Annex A: Capacity retention guarantees: maximum allowable degradation is limited to 15% over 10 years or 200,000 kilometers.",
                "Annex B: Solid-state dendrite penetration testing requires 1200 high-voltage pulsed charge cycles without internal micro-shorts.",
                "Issued under Authority Registry ECE-R100-2026. Prior iterations (2021, 2023) are designated legacy status."
            ]
        ]
    },
    {
        "filename": "iso_6469_part1_safety_2022.pdf",
        "title": "ISO 6469-1:2022 Electrically Propelled Road Vehicles - Safety Specifications",
        "doc_id": "ISO-6469-1-2022",
        "year": 2022,
        "pages": [
            [
                "ISO 6469-1:2022 International Organization for Standardization - Rechargeable Energy Storage Systems (RESS).",
                "Clause 1: Scope. Specifies electrical safety requirements for secondary battery packs mounted in wheeled electric vehicles.",
                "Clause 1.2: Overcurrent Interruption. Protection devices must sever 10 kA short-circuit current within 2 milliseconds without explosive rupture.",
                "Current sensor accuracy across dynamic discharge phases must remain within ±0.5% full-scale tolerance."
            ],
            [
                "Clause 2: Isolation Resistance Testing in Wet Conditions (2022 Methodology).",
                "Subclause 2.1: Preconditioning. The battery pack is subjected to 95% relative humidity at 40°C for 48 consecutive hours.",
                "Subclause 2.2: Measurement. Minimum insulation resistance between live components and chassis ground must exceed 500 ohms/volt.",
                "Equipotential bonding resistance between exposed conductive parts and structural vehicle ground must not exceed 0.1 ohm."
            ],
            [
                "Clause 3: Temperature Range and Thermal Runaway Containment (2022 Benchmarks).",
                "Subclause 3.1: Operational ambient temperature envelope is defined from -30°C to +55°C.",
                "Thermal management cooling fluid circuits must withstand proof pressure of 2.0 bar without micro-fissure leakage.",
                "Ethylene glycol based liquid coolant must maintain dielectric breakdown voltage greater than 25 kV."
            ],
            [
                "Clause 4: Gas Venting and Toxic Effluent Management (2022 Standards).",
                "Subclause 4.1: Emission control. In the event of single-cell venting, emissions of hydrogen fluoride (HF) must remain below 3 ppm inside the passenger cell.",
                "Carbon monoxide (CO) concentrations inside the cabin space must not exceed 50 ppm measured over an 8-hour time-weighted average.",
                "Passive pressure relief burst foils must trigger between 0.2 bar and 0.5 bar internal differential pressure."
            ],
            [
                "Clause 5: Final Inspection, Traceability, and Safety Documentation (2022 Norms).",
                "Annex C: Documentation requirements mandate delivery of Material Safety Data Sheets (MSDS) with UN 38.3 test summaries attached.",
                "Factory end-of-line testing requires high-potential (HiPot) dielectric withstand testing at 2 x Nominal Voltage + 1000 V for 60 seconds.",
                "Standard approved under ISO TC22/SC37 Committee, Revision 2022."
            ]
        ]
    },
    {
        "filename": "iso_6469_part1_safety_2025.pdf",
        "title": "ISO 6469-1:2025 Electrically Propelled Road Vehicles - Next-Gen Safety Protocols",
        "doc_id": "ISO-6469-1-2025",
        "year": 2025,
        "pages": [
            [
                "ISO 6469-1:2025 Comprehensive Revisions to Rechargeable Energy Storage Systems Safety Specifications.",
                "Clause 1: Scope Expansion. Formally incorporates bi-directional Vehicle-to-Grid (V2G) continuous cycling stress profiles.",
                "Clause 1.2: Short Circuit Protection. Pyrotechnic circuit disconnect fuses (Pyro-fuses) are made mandatory for packs exceeding 400V nominal.",
                "The disconnect actuation window under hard dead-short fault is reduced from 2.0 ms to 0.8 milliseconds."
            ],
            [
                "Clause 2: High-Voltage Isolation and Moisture Barrier Testing (2025 Updates).",
                "Subclause 2.1: Preconditioning now requires cyclic thermal shock (-40°C to +85°C) combined with salt fog mist exposure for 96 hours.",
                "Insulation resistance must remain above 1000 ohms/volt throughout dynamic charging cycles.",
                "Chassis ground bonding resistance is lowered to a maximum allowable 0.05 ohm."
            ],
            [
                "Clause 3: Active Thermal Safety and Immersion Cooling (2025 Provisions).",
                "Subclause 3.1: Immersion cooling standards. Direct contact dielectric fluids (fluorinated hydrocarbons or synthetic hydrocarbons) are codified.",
                "Dielectric fluid breakdown rating must exceed 40 kV. Flash point of active immersion coolant must be greater than 220°C.",
                "Cooling system pump redundancies require dual independent electric drive motors."
            ],
            [
                "Clause 4: Toxic Gas and Flammable Byproduct Exhaust Limits (2025 Limits).",
                "Subclause 4.1: Cabin effluent limits tightened. Hydrogen fluoride (HF) concentration must not exceed 1 ppm in passenger cabin.",
                "Cabin carbon monoxide (CO) exposure must remain below 25 ppm during any venting failure scenario.",
                "Mandatory active exhaust filtration or catalytic neutralization traps inside pack vent ducts."
            ],
            [
                "Clause 5: Quality Assurance and In-Field Telematics Reporting (2025 Rules).",
                "Annex C: Continuous cloud telematics reporting of cell impedance divergence is mandatory for fleet commercial operation.",
                "HiPot withstand voltage raised to 2.5 x Nominal Voltage + 1200 V for 60 seconds at factory QA sign-off.",
                "Standard certified under ISO TC22/SC37 Committee, Revision 2025, superseding ISO 6469-1:2022."
            ]
        ]
    },
    {
        "filename": "un383_transport_safety_rev2020.pdf",
        "title": "UN Manual of Tests and Criteria Part 38.3 Rev 2020: Transport of Lithium Batteries",
        "doc_id": "UN-383-REV2020",
        "year": 2020,
        "pages": [
            [
                "United Nations Recommendations on the Transport of Dangerous Goods: UN 38.3 Edition 2020.",
                "Section 38.3.1: Scope. Covers transport safety certification for lithium metal and lithium ion cells and batteries.",
                "All lithium ion cells must satisfy tests T.1 through T.8 prior to commercial freight shipment.",
                "Battery assemblies exceeding 12 kg gross weight are categorized as large battery systems."
            ],
            [
                "Section 38.3.2: Altitude Simulation (Test T.1) and Thermal Shock Cycling (Test T.2).",
                "T.1 Altitude: Cells are stored at an absolute pressure of 11.6 kPa (simulating 15,000 meters altitude) for a minimum of 6 hours.",
                "Criterion: No mass loss exceeding 0.1%, no leakage, no venting, and open circuit voltage retention not less than 90%.",
                "T.2 Thermal Test: 10 cycles between -40°C and +72°C with thermal transition time under 30 minutes."
            ],
            [
                "Section 38.3.3: Vibration (Test T.3) and Mechanical Shock (Test T.4) (2020 Protocols).",
                "T.3 Vibration: Sinusoidal sweep from 7 Hz to 200 Hz back to 7 Hz traversed in 15 minutes, repeated for 12 sweeps in each of 3 axes.",
                "T.4 Shock: Half-sine shock of 150g peak acceleration with pulse duration of 6 milliseconds for small cells, 50g/11ms for large packs.",
                "Pass requirement: Zero fire, zero disassembly, and voltage stability across all terminals."
            ],
            [
                "Section 38.3.4: External Short Circuit (Test T.5) and Overcharge (Test T.7).",
                "T.5 Short Circuit: External resistance less than 0.1 ohm applied at 57°C ± 4°C until case temperature returns to 57°C.",
                "Case temperature must not exceed 170°C during test T.5 for compliance under 2020 rules.",
                "T.7 Overcharge: Minimum charge current of 2x manufacturer rated continuous charge current for 24 hours."
            ],
            [
                "Section 38.3.5: State of Charge (SoC) Shipping Ceilings (2020 Regulations).",
                "Air freight restriction: Dedicated air cargo transport of standalone lithium-ion batteries requires SoC capped at 30% nominal.",
                "Maritime vessel transport allows shipments at up to 50% State of Charge under Class 9 Dangerous Goods cargo declaration.",
                "UN 38.3 Rev 2020 Summary Table issued by certified UN testing laboratory."
            ]
        ]
    },
    {
        "filename": "un383_transport_safety_rev2024.pdf",
        "title": "UN Manual of Tests and Criteria Part 38.3 Rev 2024: Maritime and Air Cargo Limits",
        "doc_id": "UN-383-REV2024",
        "year": 2024,
        "pages": [
            [
                "United Nations Manual of Tests and Criteria: Seventh Revised Edition, Amendment 2 (UN 38.3 Rev 2024).",
                "Section 38.3.1: Scope Revision. Includes updated mandates for Sodium-Ion battery chemistries alongside lithium technologies.",
                "All test regimens (T.1 to T.8) apply to sodium-ion cells exceeding 1.2V open circuit potential.",
                "Establishes strict definitions for damaged, defective, or end-of-life battery pack shipments."
            ],
            [
                "Section 38.3.2: Upgraded Thermal Test (T.2) and Overcharge Safeguards (T.7).",
                "T.2 Revision: Dwell time at extreme temperature plateaus (-40°C and +72°C) increased to 8 hours per plateau across 12 full cycles.",
                "Voltage retention baseline is raised from 90% to 92% residual open circuit potential.",
                "T.7 Overcharge testing requires secondary BMS hardware fault injection during initiation."
            ],
            [
                "Section 38.3.3: External Short Circuit Extreme Benchmark (Test T.5 Updates 2024).",
                "External short circuit resistance tightened to less than 0.05 ohm (50 milliohms).",
                "The maximum allowable cell case surface temperature is reduced from 170°C to 150°C.",
                "Test must be monitored for 6 hours post-circuit disconnection to verify delayed thermal reactions."
            ],
            [
                "Section 38.3.4: Maritime Bulk Container Transport Constraints (2024 Revision).",
                "Maritime shipping rules under the International Maritime Dangerous Goods (IMDG) Code are updated.",
                "The historical 50% maritime State of Charge allowance is formally repealed: maximum allowable maritime SoC is capped at 30%.",
                "Mandatory installation of temperature sensors logging every 15 minutes inside refrigerated shipping containers (reefers)."
            ],
            [
                "Section 38.3.5: Damaged / Recycled Cell Freight Safety Protocols (2024 Rules).",
                "Damaged or defective batteries (SP 376) must be transported in non-combustible vermiculite packaging rated for pyro-containment.",
                "Packaging must withstand an internal deflagration temperature of 1000°C for 30 minutes without external case breach.",
                "Standard validated under UN Dangerous Goods Transport Subcommittee Resolution 2024."
            ]
        ]
    },
    {
        "filename": "us_dot_nhtsa_battery_fmvss_2022.pdf",
        "title": "NHTSA FMVSS 305 Rev 2022: Electric Vehicle Crash and Electrical Spill Rules",
        "doc_id": "NHTSA-FMVSS305-2022",
        "year": 2022,
        "pages": [
            [
                "US Department of Transportation: NHTSA Federal Motor Vehicle Safety Standard (FMVSS) No. 305 (2022 Enforcement).",
                "1. Purpose: Mitigate fatalities and injuries from electrical shock and electrolyte burn hazards during and after motor vehicle collisions.",
                "Application: All passenger cars, multipurpose vehicles, and light trucks with electrical components operating above 60 V DC.",
                "Standard barrier impact crashes executed at 35 mph (56 km/h) into fixed frontal concrete barriers."
            ],
            [
                "2. Electrolyte Spillage Limitations (FMVSS 305-2022 Thresholds).",
                "Clause 2.1: Post-crash electrolyte spillage outside the vehicle must not exceed 5.0 liters within the first 30 minutes post-impact.",
                "Inside the passenger compartment, zero liquid electrolyte leakage is permitted under any collision orientation.",
                "Spillage collection trays must be installed beneath battery structures during dynamic testing."
            ],
            [
                "3. Battery Enclosure Retention and Mechanical Attachment (2022 Criteria).",
                "Clause 3.1: Retention. The battery assembly must remain anchored to the vehicle chassis rails.",
                "No mounting bracket failure or complete structural separation from floor pan cross-members is allowed under side-pole impact (20 mph).",
                "Intrusion of chassis rails into cell modules must not exceed 20 mm."
            ],
            [
                "4. Electrical Isolation Verification Post-Impact (2022 Standards).",
                "Clause 4.1: Post-collision electrical isolation of high-voltage DC busbars must be greater than 500 ohms per volt.",
                "Alternatively, if high voltage isolation is lost, voltage on all exposed cables must discharge below 60 V DC within 5.0 seconds.",
                "Capacitive energy storage retention on disconnected lines must drop below 0.2 Joules."
            ],
            [
                "5. Administrative Certification and Compliance Filings (2022 Code).",
                "Section 5: Manufacturers must submit dynamic test crash telemetry to NHTSA Office of Vehicle Safety Compliance within 60 days.",
                "All vehicles must feature an easily accessible manual First Responder Loop physically severing high-voltage contactor circuits.",
                "FMVSS 305 Rev 2022 certified by US Department of Transportation."
            ]
        ]
    },
    {
        "filename": "us_dot_nhtsa_battery_fmvss_2025.pdf",
        "title": "NHTSA FMVSS 305 Rev 2025: Severe Angle Crash and Fire Suppression Rules",
        "doc_id": "NHTSA-FMVSS305-2025",
        "year": 2025,
        "pages": [
            [
                "US Department of Transportation: NHTSA FMVSS No. 305 Revisions and High-Energy Safety Enhancements (2025).",
                "1. Scope Expansion: Incorporates commercial medium and heavy-duty vehicles (Class 4 through Class 8 electric trucks).",
                "Frontal impact speed test requirement raised from 35 mph to 40 mph (64 km/h) against 50% overlap deformable barriers.",
                "Side impact pole velocity increased to 25 mph with mandatory oblique angle (75-degree) impact verification."
            ],
            [
                "2. Zero-Tolerance Electrolyte Discharge Rules (2025 Overhaul).",
                "Clause 2.1: The 2022 allowance of 5.0 liters external electrolyte leakage is eliminated.",
                "Under the 2025 standard, maximum allowable electrolyte spillage is exactly 0.0 liters (zero discharge permitted).",
                "Any liquid detection outside battery pack seals constitutes an automatic compliance failure."
            ],
            [
                "3. Mandatory High-Voltage Discharge Acceleration (2025 Speed Requirements).",
                "Clause 3.1: Active discharge mechanism. High-voltage bus residual charge must discharge below 60 V DC in less than 2.0 seconds.",
                "This updates the prior 5.0-second allowance from the 2022 regulation to protect trapped occupants and emergency rescue workers.",
                "Energy stored in DC filter capacitors must dissipate below 0.1 Joules within the 2.0-second window."
            ],
            [
                "4. Active Pyrotechnic Fire Suppression Systems (2025 Innovation Mandate).",
                "Clause 4.1: Traction batteries exceeding 80 kWh must integrate internal active fire extinguishing or aerosol suppression systems.",
                "Suppression systems must flood battery cell compartments with potassium aerosol or clean gas within 500 milliseconds of thermal runaway detection.",
                "Containment system must prevent reignition for at least 60 minutes post-impact."
            ],
            [
                "5. Safety Filing and Automated Telematics Registry (2025 Requirements).",
                "Section 5: Real-time automatic collision notification (eCall) must transmit battery high-voltage disconnection confirmation to 911 dispatchers.",
                "Annual OEM recertification testing required for every battery pack revision altering cell chemistry or structural casing.",
                "Promulgated by NHTSA Executive Directive, Standard FMVSS 305-2025."
            ]
        ]
    },
    {
        "filename": "eu_battery_directive_sustainability_2024.pdf",
        "title": "EU Battery Regulation 2024/1542: Carbon Footprint and Due Diligence",
        "doc_id": "EU-BAT-2024",
        "year": 2024,
        "pages": [
            [
                "Regulation (EU) 2024/1542 of the European Parliament and of the Council concerning Batteries and Waste Batteries.",
                "Chapter I: General Provisions and Scope. Applies to all categories of batteries: portable, starting/lighting, and electric vehicle batteries.",
                "Article 1: Sets targets for carbon footprint declarations, recycled content thresholds, and performance durability criteria across member states.",
                "By August 2024, all EV batteries entering the European market must possess a verified Carbon Footprint Declaration."
            ],
            [
                "Chapter II: Carbon Footprint Calculation and Maximum Lifecycle Limits.",
                "Article 7: Calculation methodology encompasses upstream raw material acquisition, transport, cell manufacturing, and pack assembly.",
                "Carbon intensity is reported as kilograms of CO2 equivalent per kilowatt-hour of total battery storage capacity (kg CO2e/kWh).",
                "Starting December 2024, batteries exceeding the third-party verified maximum threshold of 110 kg CO2e/kWh cannot be registered."
            ],
            [
                "Chapter III: Mandatory Minimum Recycled Material Contents (2024 Milestones).",
                "Article 8: Minimum shares of recovered material present in active battery materials are legally binding.",
                "Mandatory targets: 16% recovered Cobalt, 85% recovered Lead, 6% recovered Lithium, and 6% recovered Nickel.",
                "Manufacturers must maintain chain-of-custody documentation certified via European Union notified inspection bodies."
            ],
            [
                "Chapter IV: Removability, Replaceability, and Battery Health Transparency.",
                "Article 11: Electric vehicle batteries must allow diagnosis and module-level repair by certified independent operators.",
                "Battery Management System telemetry must provide open API access for the user to query State of Certified Energy (SoCE).",
                "Cell degradation telemetry must be preserved in permanent flash memory for vehicle operational lifespan."
            ],
            [
                "Chapter V: The Digital European Battery Passport (Implementation Framework).",
                "Article 65: Mandatory implementation date set for February 2027, with pilot data registries commencing registration in 2024.",
                "Passport must store cell chemistry details, manufacturing location, recycled material percentages, and full carbon footprint audit.",
                "Enforced across all 27 EU Member States under Official Journal of the European Union Directive 2024/1542."
            ]
        ]
    }
]

def generate_pdf(doc_spec: dict):
    doc = pymupdf.open()
    
    # 5 Pages minimum
    for page_idx, page_paragraphs in enumerate(doc_spec["pages"]):
        page = doc.new_page(width=595, height=842) # A4 format
        
        # Header block
        header_text = f"DOCUMENT: {doc_spec['filename']} | REGULATION CODE: {doc_spec['doc_id']} | YEAR: {doc_spec['year']}"
        page.insert_textbox(pymupdf.Rect(50, 30, 545, 50), header_text, fontsize=8, fontname="helv", color=(0.4, 0.4, 0.4))
        
        # Title block on Page 1
        y_cursor = 60
        if page_idx == 0:
            page.insert_textbox(pymupdf.Rect(50, y_cursor, 545, y_cursor + 45), doc_spec["title"], fontsize=14, fontname="hebo")
            y_cursor += 55
            
        # Body Paragraphs
        for p in page_paragraphs:
            rect = pymupdf.Rect(50, y_cursor, 545, y_cursor + 90)
            page.insert_textbox(rect, p, fontsize=10.5, fontname="helv", lineheight=1.3)
            y_cursor += 95
            
        # Footer block (Pagination)
        footer_text = f"Page {page_idx + 1} of {len(doc_spec['pages'])} | Standard Controlled Copy - Compliance Registry"
        page.insert_textbox(pymupdf.Rect(50, 800, 545, 820), footer_text, fontsize=8, fontname="helv", color=(0.5, 0.5, 0.5))

    out_path = docs_dir / doc_spec["filename"]
    doc.save(out_path)
    doc.close()
    print(f"Generated ({len(doc_spec['pages'])} pages): {out_path.name}")

def main():
    print("=" * 70)
    print("GENERATING 10 DETAILED MULTI-PAGE REGULATORY PDF DOCUMENTS (≥ 5 PAGES EACH)")
    print("=" * 70)
    
    # Clean previous test files in documents/
    for existing_file in docs_dir.glob("*.pdf"):
        existing_file.unlink()
        
    for doc_spec in DOCS_DATA:
        generate_pdf(doc_spec)
        
    print("=" * 70)
    print("All 10 documents generated successfully in ./documents/")

if __name__ == "__main__":
    main()