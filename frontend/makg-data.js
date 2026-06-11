// makg-data.js — Mock knowledge graph data for MaKG
// Biomedical / SciERC domain: metabolic conditions, biomarkers, methods
window.MAKG_DATA = (function () {

  const ENTITY_TYPES = {
    METABOLIC_CONDITION: { color: '#2e7d9e', label: 'Metabolic Condition' },
    BIOMARKER:           { color: '#c17f4a', label: 'Biomarker' },
    CLINICAL_MEASURE:    { color: '#7264c8', label: 'Clinical Measure' },
    METHOD:              { color: '#3d8f6e', label: 'Method' },
    TASK:                { color: '#a05070', label: 'Task' },
    MATERIAL:            { color: '#4a6b9e', label: 'Material' },
  };

  const ENTITIES = [
    // ── METABOLIC CONDITIONS ────────────────────────────────────────────────
    { id: 'type2_diabetes',          labels: ['Type 2 Diabetes', 'T2DM'],                    type: 'METABOLIC_CONDITION', metadata: { confidence: 0.97, source: 'PubMed:PMC7234', description: 'Chronic metabolic disorder characterized by insulin resistance and hyperglycemia' } },
    { id: 'insulin_resistance',      labels: ['Insulin Resistance', 'IR'],                   type: 'METABOLIC_CONDITION', metadata: { confidence: 0.95, source: 'PubMed:PMC8821', description: 'Reduced cellular response to insulin signaling in peripheral tissues' } },
    { id: 'metabolic_syndrome',      labels: ['Metabolic Syndrome', 'MetS'],                 type: 'METABOLIC_CONDITION', metadata: { confidence: 0.93, source: 'PubMed:PMC6612', description: 'Cluster of conditions increasing risk of cardiovascular disease and T2DM' } },
    { id: 'nonalcoholic_fatty_liver',labels: ['NAFLD', 'Non-Alcoholic Fatty Liver Disease'], type: 'METABOLIC_CONDITION', metadata: { confidence: 0.91, source: 'PubMed:PMC7890', description: 'Hepatic fat accumulation not caused by alcohol consumption' } },
    { id: 'obesity',                 labels: ['Obesity', 'Adiposity'],                       type: 'METABOLIC_CONDITION', metadata: { confidence: 0.96, source: 'PubMed:PMC5521', description: 'Excess body fat accumulation with adverse metabolic health effects' } },

    // ── BIOMARKERS ──────────────────────────────────────────────────────────
    { id: 'hba1c',          labels: ['HbA1c', 'Glycated Hemoglobin'],        type: 'BIOMARKER', metadata: { confidence: 0.98, source: 'ADA:Standards-2023', description: 'Reflects average blood glucose over the past 2–3 months' } },
    { id: 'fasting_glucose',labels: ['Fasting Blood Glucose', 'FBG'],        type: 'BIOMARKER', metadata: { confidence: 0.97, source: 'ADA:Standards-2023', description: 'Plasma glucose concentration after a minimum 8-hour fast' } },
    { id: 'adiponectin',    labels: ['Adiponectin', 'ADIPOQ'],               type: 'BIOMARKER', metadata: { confidence: 0.89, source: 'PubMed:PMC6723', description: 'Adipokine with insulin-sensitizing and anti-inflammatory properties' } },
    { id: 'il6',            labels: ['Interleukin-6', 'IL-6'],               type: 'BIOMARKER', metadata: { confidence: 0.94, source: 'PubMed:PMC8812', description: 'Pro-inflammatory cytokine elevated in metabolic disorders' } },
    { id: 'crp',            labels: ['C-Reactive Protein', 'CRP'],           type: 'BIOMARKER', metadata: { confidence: 0.96, source: 'PubMed:PMC5534', description: 'Systemic inflammation marker produced by the liver in response to IL-6' } },
    { id: 'insulin_level',  labels: ['Serum Insulin', 'Fasting Insulin'],    type: 'BIOMARKER', metadata: { confidence: 0.95, source: 'PubMed:PMC4415', description: 'Circulating insulin concentration used in HOMA-IR calculation' } },

    // ── CLINICAL MEASURES ───────────────────────────────────────────────────
    { id: 'bmi',                labels: ['BMI', 'Body Mass Index'],                    type: 'CLINICAL_MEASURE', metadata: { confidence: 0.99, source: 'WHO:ICD-11', description: 'Weight-to-height² ratio; ≥30 kg/m² defines obesity' } },
    { id: 'waist_circumference', labels: ['Waist Circumference', 'WC'],               type: 'CLINICAL_MEASURE', metadata: { confidence: 0.97, source: 'WHO:ICD-11', description: 'Abdominal obesity proxy; >102 cm (M) or >88 cm (F) is clinically significant' } },
    { id: 'homa_ir',            labels: ['HOMA-IR', 'Homeostatic Model Assessment'],  type: 'CLINICAL_MEASURE', metadata: { confidence: 0.93, source: 'PubMed:PMC7812', description: 'Composite insulin resistance index = (fasting glucose × fasting insulin) / 22.5' } },
    { id: 'ogtt',               labels: ['OGTT', 'Oral Glucose Tolerance Test'],      type: 'CLINICAL_MEASURE', metadata: { confidence: 0.94, source: 'ADA:Standards-2023', description: '75g oral glucose challenge with 2-hour plasma glucose measurement' } },

    // ── METHODS ─────────────────────────────────────────────────────────────
    { id: 'gc_ms',           labels: ['GC-MS', 'Gas Chromatography–Mass Spectrometry'], type: 'METHOD', metadata: { confidence: 0.92, source: 'Methods:PMC6234', description: 'Untargeted metabolomics platform for small-molecule profiling in biofluids' } },
    { id: 'rna_seq',         labels: ['RNA-seq', 'RNA Sequencing'],                     type: 'METHOD', metadata: { confidence: 0.95, source: 'Methods:PMC7654', description: 'Transcriptome-wide gene expression quantification via next-gen sequencing' } },
    { id: 'gwas',            labels: ['GWAS', 'Genome-Wide Association Study'],         type: 'METHOD', metadata: { confidence: 0.97, source: 'Methods:PMC8234', description: 'Population-level genotype–phenotype association mapping' } },
    { id: 'elisa',           labels: ['ELISA', 'Enzyme-Linked Immunosorbent Assay'],   type: 'METHOD', metadata: { confidence: 0.98, source: 'Methods:PMC3421', description: 'Antibody-based protein quantification in biological fluids' } },
    { id: 'fasting_protocol',labels: ['Fasting Protocol', '8h Overnight Fast'],         type: 'METHOD', metadata: { confidence: 0.91, source: 'Protocol:PMC5512', description: 'Standardized pre-analytical procedure for metabolic and hormonal assays' } },

    // ── TASKS ───────────────────────────────────────────────────────────────
    { id: 'glycemic_control',   labels: ['Glycemic Control', 'Blood Sugar Management'],    type: 'TASK', metadata: { confidence: 0.96, source: 'ADA:Standards-2023', description: 'Clinical objective of maintaining normoglycemia in T2DM patients' } },
    { id: 'biomarker_discovery',labels: ['Biomarker Discovery', 'Biomarker Identification'], type: 'TASK', metadata: { confidence: 0.88, source: 'Research:PMC7723', description: 'Systematic identification of disease-specific molecular markers' } },
    { id: 'risk_stratification',labels: ['Risk Stratification', 'Patient Risk Assessment'], type: 'TASK', metadata: { confidence: 0.90, source: 'Clinical:PMC6523', description: 'Categorization of patients by metabolic disease risk level' } },
    { id: 'pathway_analysis',   labels: ['Pathway Analysis', 'Metabolic Pathway Investigation'], type: 'TASK', metadata: { confidence: 0.87, source: 'Research:PMC8123', description: 'Systems-level mapping of dysregulated biological pathways' } },
    { id: 'clinical_validation',labels: ['Clinical Validation', 'Biomarker Validation'],    type: 'TASK', metadata: { confidence: 0.92, source: 'Clinical:PMC7212', description: 'Prospective validation of biomarker performance in clinical cohorts' } },

    // ── MATERIALS ───────────────────────────────────────────────────────────
    { id: 'blood_plasma',      labels: ['Blood Plasma', 'Plasma Sample'],                       type: 'MATERIAL', metadata: { confidence: 0.99, source: 'Protocol:STANDARD', description: 'Liquid fraction of blood after centrifugation; primary sample for protein biomarkers' } },
    { id: 'adipose_tissue',    labels: ['Adipose Tissue', 'Fat Tissue'],                        type: 'MATERIAL', metadata: { confidence: 0.95, source: 'Anatomy:Standard', description: 'Connective tissue composed of adipocytes; major site of adipokine secretion' } },
    { id: 'liver_biopsy',      labels: ['Liver Biopsy', 'Hepatic Tissue'],                      type: 'MATERIAL', metadata: { confidence: 0.93, source: 'Protocol:PMC5621', description: 'Gold standard for NAFLD histological staging (NAS score)' } },
    { id: 'peripheral_blood',  labels: ['PBMCs', 'Peripheral Blood Mononuclear Cells'],          type: 'MATERIAL', metadata: { confidence: 0.94, source: 'Protocol:PMC6821', description: 'Lymphocytes and monocytes isolated for GWAS and immune profiling' } },
    { id: 'urine_sample',      labels: ['Urine Sample', 'Urine'],                               type: 'MATERIAL', metadata: { confidence: 0.96, source: 'Protocol:STANDARD', description: 'Non-invasive biofluid ideal for untargeted metabolomics profiling' } },
  ];

  const TRIPLES = [
    // T2DM connections
    { id: 't1',  subject: 'type2_diabetes',           relation: 'IS_ASSOCIATED_WITH',  object: 'insulin_resistance',       confidence: 0.97, source: 'PubMed:PMC8821' },
    { id: 't2',  subject: 'type2_diabetes',           relation: 'HAS_BIOMARKER',        object: 'hba1c',                    confidence: 0.98, source: 'ADA:Standards-2023' },
    { id: 't3',  subject: 'type2_diabetes',           relation: 'HAS_BIOMARKER',        object: 'fasting_glucose',          confidence: 0.97, source: 'ADA:Standards-2023' },
    { id: 't4',  subject: 'type2_diabetes',           relation: 'CHARACTERIZED_BY',     object: 'metabolic_syndrome',       confidence: 0.88, source: 'PubMed:PMC6612' },
    // Insulin resistance
    { id: 't5',  subject: 'insulin_resistance',       relation: 'MEASURED_BY',          object: 'homa_ir',                  confidence: 0.93, source: 'PubMed:PMC7812' },
    { id: 't6',  subject: 'insulin_resistance',       relation: 'INCREASES_LEVEL_OF',   object: 'il6',                      confidence: 0.85, source: 'PubMed:PMC8812' },
    { id: 't7',  subject: 'insulin_resistance',       relation: 'DECREASES_LEVEL_OF',   object: 'adiponectin',              confidence: 0.87, source: 'PubMed:PMC6723' },
    // Metabolic syndrome
    { id: 't8',  subject: 'metabolic_syndrome',       relation: 'INCLUDES_CRITERION',   object: 'bmi',                      confidence: 0.92, source: 'WHO:ICD-11' },
    { id: 't9',  subject: 'metabolic_syndrome',       relation: 'INCLUDES_CRITERION',   object: 'waist_circumference',      confidence: 0.91, source: 'WHO:ICD-11' },
    { id: 't10', subject: 'metabolic_syndrome',       relation: 'CO_OCCURS_WITH',       object: 'nonalcoholic_fatty_liver',  confidence: 0.86, source: 'PubMed:PMC7890' },
    // NAFLD
    { id: 't11', subject: 'nonalcoholic_fatty_liver', relation: 'ASSESSED_VIA',         object: 'liver_biopsy',             confidence: 0.94, source: 'PubMed:PMC7890' },
    { id: 't12', subject: 'nonalcoholic_fatty_liver', relation: 'IS_ASSOCIATED_WITH',   object: 'obesity',                  confidence: 0.89, source: 'PubMed:PMC5521' },
    // Obesity
    { id: 't13', subject: 'obesity',                  relation: 'MEASURED_BY',          object: 'bmi',                      confidence: 0.99, source: 'WHO:ICD-11' },
    { id: 't14', subject: 'obesity',                  relation: 'MEASURED_BY',          object: 'waist_circumference',      confidence: 0.95, source: 'WHO:ICD-11' },
    // HbA1c
    { id: 't15', subject: 'hba1c',                    relation: 'QUANTIFIED_BY',        object: 'elisa',                    confidence: 0.96, source: 'Methods:PMC3421' },
    { id: 't16', subject: 'hba1c',                    relation: 'MEASURED_IN',          object: 'blood_plasma',             confidence: 0.97, source: 'Protocol:STANDARD' },
    // Fasting glucose
    { id: 't17', subject: 'fasting_glucose',          relation: 'REQUIRES_PROTOCOL',    object: 'fasting_protocol',         confidence: 0.91, source: 'ADA:Standards-2023' },
    { id: 't18', subject: 'fasting_glucose',          relation: 'MEASURED_IN',          object: 'blood_plasma',             confidence: 0.98, source: 'Protocol:STANDARD' },
    // Adiponectin
    { id: 't19', subject: 'adiponectin',              relation: 'EXPRESSED_IN',         object: 'adipose_tissue',           confidence: 0.93, source: 'Anatomy:Standard' },
    { id: 't20', subject: 'adiponectin',              relation: 'QUANTIFIED_BY',        object: 'elisa',                    confidence: 0.95, source: 'Methods:PMC3421' },
    // IL-6
    { id: 't21', subject: 'il6',                      relation: 'MEASURED_IN',          object: 'blood_plasma',             confidence: 0.95, source: 'Protocol:STANDARD' },
    { id: 't22', subject: 'il6',                      relation: 'UPREGULATED_IN',       object: 'adipose_tissue',           confidence: 0.88, source: 'PubMed:PMC8812' },
    // CRP
    { id: 't23', subject: 'crp',                      relation: 'INDICATES',            object: 'metabolic_syndrome',       confidence: 0.87, source: 'PubMed:PMC5534' },
    { id: 't24', subject: 'crp',                      relation: 'QUANTIFIED_BY',        object: 'elisa',                    confidence: 0.97, source: 'Methods:PMC3421' },
    { id: 't25', subject: 'crp',                      relation: 'MEASURED_IN',          object: 'blood_plasma',             confidence: 0.96, source: 'Protocol:STANDARD' },
    // Serum insulin
    { id: 't26', subject: 'insulin_level',            relation: 'QUANTIFIED_BY',        object: 'elisa',                    confidence: 0.97, source: 'Methods:PMC3421' },
    { id: 't27', subject: 'insulin_level',            relation: 'MEASURED_IN',          object: 'blood_plasma',             confidence: 0.96, source: 'Protocol:STANDARD' },
    // HOMA-IR
    { id: 't28', subject: 'homa_ir',                  relation: 'DERIVED_FROM',         object: 'fasting_glucose',          confidence: 0.99, source: 'Clinical:PMC7812' },
    { id: 't29', subject: 'homa_ir',                  relation: 'DERIVED_FROM',         object: 'insulin_level',            confidence: 0.99, source: 'Clinical:PMC7812' },
    // OGTT
    { id: 't30', subject: 'ogtt',                     relation: 'USED_IN',              object: 'clinical_validation',      confidence: 0.91, source: 'ADA:Standards-2023' },
    { id: 't31', subject: 'ogtt',                     relation: 'MEASURES',             object: 'fasting_glucose',          confidence: 0.93, source: 'ADA:Standards-2023' },
    // GC-MS
    { id: 't32', subject: 'gc_ms',                    relation: 'APPLIED_TO',           object: 'urine_sample',             confidence: 0.90, source: 'Methods:PMC6234' },
    { id: 't33', subject: 'gc_ms',                    relation: 'ENABLES',              object: 'biomarker_discovery',      confidence: 0.88, source: 'Research:PMC7723' },
    // RNA-seq
    { id: 't34', subject: 'rna_seq',                  relation: 'APPLIED_TO',           object: 'adipose_tissue',           confidence: 0.93, source: 'Methods:PMC7654' },
    { id: 't35', subject: 'rna_seq',                  relation: 'ENABLES',              object: 'pathway_analysis',         confidence: 0.92, source: 'Research:PMC8123' },
    // GWAS
    { id: 't36', subject: 'gwas',                     relation: 'IDENTIFIES',           object: 'type2_diabetes',           confidence: 0.89, source: 'Methods:PMC8234' },
    { id: 't37', subject: 'gwas',                     relation: 'APPLIED_TO',           object: 'peripheral_blood',         confidence: 0.95, source: 'Protocol:PMC6821' },
    // Biomarker discovery
    { id: 't38', subject: 'biomarker_discovery',      relation: 'REQUIRES',             object: 'gc_ms',                    confidence: 0.88, source: 'Research:PMC7723' },
    { id: 't39', subject: 'biomarker_discovery',      relation: 'VALIDATED_BY',         object: 'clinical_validation',      confidence: 0.86, source: 'Clinical:PMC7212' },
    // Risk stratification
    { id: 't40', subject: 'risk_stratification',      relation: 'USES',                 object: 'hba1c',                    confidence: 0.93, source: 'Clinical:PMC6523' },
    { id: 't41', subject: 'risk_stratification',      relation: 'USES',                 object: 'bmi',                      confidence: 0.91, source: 'Clinical:PMC6523' },
    // Pathway analysis
    { id: 't42', subject: 'pathway_analysis',         relation: 'INVESTIGATES',         object: 'insulin_resistance',       confidence: 0.87, source: 'Research:PMC8123' },
    // Clinical validation
    { id: 't43', subject: 'clinical_validation',      relation: 'REQUIRES_MATERIAL',    object: 'blood_plasma',             confidence: 0.94, source: 'Protocol:STANDARD' },
    // Tissue relationships
    { id: 't44', subject: 'peripheral_blood',         relation: 'CONTAINS',             object: 'il6',                      confidence: 0.93, source: 'Anatomy:Standard' },
    { id: 't45', subject: 'adipose_tissue',           relation: 'IS_SOURCE_OF',         object: 'adiponectin',              confidence: 0.95, source: 'Anatomy:Standard' },
  ];

  const SAMPLE_QA = [
    {
      id: 'qa1',
      question: 'Which biomarkers are elevated in metabolic syndrome and how are they measured?',
      answer: `Metabolic syndrome presents a distinct inflammatory and metabolic biomarker signature. **C-Reactive Protein (CRP)** is a systemic inflammation marker consistently elevated in MetS, quantified via ELISA immunoassay from blood plasma. **Interleukin-6 (IL-6)**, upregulated in adipose tissue, drives hepatic CRP synthesis and is measurable from blood plasma.\n\nInsulin resistance—central to metabolic syndrome—is assessed via the **HOMA-IR score**, a composite derived from fasting blood glucose and serum insulin (both ELISA-quantified). Anthropometric criteria **BMI** (≥30 kg/m²) and **waist circumference** (>102 cm men / >88 cm women) constitute WHO diagnostic criteria for MetS.`,
      referencedNodes: new Set(['crp', 'il6', 'insulin_level', 'metabolic_syndrome', 'elisa', 'blood_plasma', 'homa_ir', 'fasting_glucose', 'bmi', 'waist_circumference', 'adipose_tissue']),
      referencedEdges: new Set(['t23', 't24', 't25', 't21', 't22', 't28', 't29', 't8', 't9', 't26', 't27']),
      model: 'GPT-4o', timestamp: '09:14', nodeCount: 11,
    },
    {
      id: 'qa2',
      question: 'What methods are used to study type 2 diabetes biomarkers in tissue samples?',
      answer: `Multiple methodological approaches are deployed across tissue types to profile T2DM biomarkers:\n\n**GC-MS metabolomics** applied to urine samples enables untargeted small-molecule biomarker discovery. **RNA sequencing** of adipose tissue reveals transcriptional regulation of insulin sensitivity genes. **GWAS** using peripheral blood mononuclear cells identifies genetic variants conferring T2DM susceptibility. **ELISA** remains the gold standard for quantifying circulating markers—HbA1c, adiponectin, fasting glucose—from blood plasma.\n\nThese platforms are increasingly combined in multi-omics validation pipelines for robust biomarker discovery and clinical validation.`,
      referencedNodes: new Set(['gc_ms', 'rna_seq', 'gwas', 'elisa', 'urine_sample', 'adipose_tissue', 'peripheral_blood', 'blood_plasma', 'hba1c', 'adiponectin', 'type2_diabetes', 'fasting_glucose', 'biomarker_discovery']),
      referencedEdges: new Set(['t32', 't33', 't34', 't35', 't36', 't37', 't15', 't16', 't18', 't20', 't38']),
      model: 'GPT-5', timestamp: '10:02', nodeCount: 13,
    },
    {
      id: 'qa3',
      question: 'How is insulin resistance measured and what downstream conditions does it drive?',
      answer: `Insulin resistance is quantified primarily via the **HOMA-IR score** (fasting glucose × fasting insulin / 22.5), with both inputs measured by ELISA from blood plasma. The **Oral Glucose Tolerance Test (OGTT)** provides dynamic glucose disposal assessment for formal clinical validation.\n\nInsulin resistance mechanistically drives several downstream conditions: **Type 2 Diabetes** through chronic impaired glucose uptake, **NAFLD** via dysregulated hepatic lipid metabolism, and **Metabolic Syndrome** as a core pathophysiological component. Molecularly, it elevates circulating **IL-6** and suppresses **adiponectin** from adipose tissue. Risk stratification incorporates **HbA1c** and **BMI** as prognostic indicators.`,
      referencedNodes: new Set(['insulin_resistance', 'homa_ir', 'fasting_glucose', 'insulin_level', 'ogtt', 'type2_diabetes', 'nonalcoholic_fatty_liver', 'metabolic_syndrome', 'il6', 'adiponectin', 'elisa', 'blood_plasma', 'hba1c', 'bmi']),
      referencedEdges: new Set(['t5', 't6', 't7', 't28', 't29', 't30', 't31', 't1', 't10', 't12', 't40', 't41', 't15', 't16', 't26', 't27']),
      model: 'DeepSeek-v3', timestamp: '10:45', nodeCount: 14,
    },
  ];

  const RELATION_TYPES = [...new Set(TRIPLES.map(t => t.relation))];

  return { ENTITY_TYPES, ENTITIES, TRIPLES, SAMPLE_QA, RELATION_TYPES };
})();
