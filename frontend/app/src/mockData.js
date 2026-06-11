/**
 * Mock KG data — used when the API server is unreachable.
 * Matches the schema returned by GET /kg/data.
 */

export const MOCK_ENTITY_TYPES = {
  METABOLIC_CONDITION: { color: '#2e7d9e', label: 'Metabolic Condition' },
  BIOMARKER:           { color: '#c17f4a', label: 'Biomarker' },
  CLINICAL_MEASURE:    { color: '#7264c8', label: 'Clinical Measure' },
  METHOD:              { color: '#3d8f6e', label: 'Method' },
  TASK:                { color: '#a05070', label: 'Task' },
  MATERIAL:            { color: '#4a6b9e', label: 'Material' },
};

export const MOCK_ENTITIES = [
  { id: 'type2_diabetes',      labels: ['Type 2 Diabetes', 'T2DM'],           type: 'METABOLIC_CONDITION', metadata: { confidence: 0.97, source: 'PubMed:PMC7234', description: 'Chronic metabolic disorder characterized by insulin resistance and hyperglycemia' } },
  { id: 'insulin_resistance',  labels: ['Insulin Resistance', 'IR'],          type: 'METABOLIC_CONDITION', metadata: { confidence: 0.95, source: 'PubMed:PMC8821', description: 'Reduced cellular response to insulin signaling' } },
  { id: 'metabolic_syndrome',  labels: ['Metabolic Syndrome', 'MetS'],        type: 'METABOLIC_CONDITION', metadata: { confidence: 0.93, source: 'PubMed:PMC6612', description: 'Cluster of conditions increasing CVD and T2DM risk' } },
  { id: 'nafld',               labels: ['NAFLD'],                             type: 'METABOLIC_CONDITION', metadata: { confidence: 0.91, source: 'PubMed:PMC7890', description: 'Hepatic fat accumulation not caused by alcohol' } },
  { id: 'obesity',             labels: ['Obesity', 'Adiposity'],              type: 'METABOLIC_CONDITION', metadata: { confidence: 0.96, source: 'PubMed:PMC5521', description: 'Excess body fat accumulation' } },
  { id: 'hba1c',               labels: ['HbA1c', 'Glycated Hemoglobin'],     type: 'BIOMARKER',           metadata: { confidence: 0.98, source: 'ADA:Standards-2023', description: 'Reflects average blood glucose over 2-3 months' } },
  { id: 'fasting_glucose',     labels: ['Fasting Blood Glucose', 'FBG'],     type: 'BIOMARKER',           metadata: { confidence: 0.97, source: 'ADA:Standards-2023', description: 'Plasma glucose after 8-hour fast' } },
  { id: 'adiponectin',         labels: ['Adiponectin', 'ADIPOQ'],            type: 'BIOMARKER',           metadata: { confidence: 0.89, source: 'PubMed:PMC6723', description: 'Insulin-sensitizing adipokine' } },
  { id: 'il6',                 labels: ['Interleukin-6', 'IL-6'],            type: 'BIOMARKER',           metadata: { confidence: 0.94, source: 'PubMed:PMC8812', description: 'Pro-inflammatory cytokine' } },
  { id: 'crp',                 labels: ['C-Reactive Protein', 'CRP'],        type: 'BIOMARKER',           metadata: { confidence: 0.96, source: 'PubMed:PMC5534', description: 'Systemic inflammation marker' } },
  { id: 'insulin_level',       labels: ['Serum Insulin'],                    type: 'BIOMARKER',           metadata: { confidence: 0.95, source: 'PubMed:PMC4415', description: 'Circulating insulin concentration' } },
  { id: 'bmi',                 labels: ['BMI', 'Body Mass Index'],           type: 'CLINICAL_MEASURE',    metadata: { confidence: 0.99, source: 'WHO:ICD-11', description: 'Weight-to-height ratio' } },
  { id: 'waist_circumference', labels: ['Waist Circumference', 'WC'],       type: 'CLINICAL_MEASURE',    metadata: { confidence: 0.97, source: 'WHO:ICD-11', description: 'Abdominal obesity proxy' } },
  { id: 'homa_ir',             labels: ['HOMA-IR'],                          type: 'CLINICAL_MEASURE',    metadata: { confidence: 0.93, source: 'PubMed:PMC7812', description: 'Insulin resistance index' } },
  { id: 'ogtt',                labels: ['OGTT'],                             type: 'CLINICAL_MEASURE',    metadata: { confidence: 0.94, source: 'ADA:Standards-2023', description: 'Oral glucose tolerance test' } },
  { id: 'gc_ms',               labels: ['GC-MS'],                            type: 'METHOD',              metadata: { confidence: 0.92, source: 'Methods:PMC6234', description: 'Metabolomics platform' } },
  { id: 'rna_seq',             labels: ['RNA-seq'],                          type: 'METHOD',              metadata: { confidence: 0.95, source: 'Methods:PMC7654', description: 'Gene expression quantification' } },
  { id: 'gwas',                labels: ['GWAS'],                             type: 'METHOD',              metadata: { confidence: 0.97, source: 'Methods:PMC8234', description: 'Genetic association study' } },
  { id: 'elisa',               labels: ['ELISA'],                            type: 'METHOD',              metadata: { confidence: 0.98, source: 'Methods:PMC3421', description: 'Protein quantification assay' } },
  { id: 'biomarker_discovery', labels: ['Biomarker Discovery'],              type: 'TASK',                metadata: { confidence: 0.88, source: 'Research:PMC7723', description: 'Systematic marker identification' } },
  { id: 'risk_stratification', labels: ['Risk Stratification'],              type: 'TASK',                metadata: { confidence: 0.90, source: 'Clinical:PMC6523', description: 'Patient risk assessment' } },
  { id: 'pathway_analysis',    labels: ['Pathway Analysis'],                 type: 'TASK',                metadata: { confidence: 0.87, source: 'Research:PMC8123', description: 'Metabolic pathway investigation' } },
  { id: 'blood_plasma',        labels: ['Blood Plasma'],                     type: 'MATERIAL',            metadata: { confidence: 0.99, source: 'Protocol:STANDARD', description: 'Primary biofluid sample' } },
  { id: 'adipose_tissue',      labels: ['Adipose Tissue'],                   type: 'MATERIAL',            metadata: { confidence: 0.95, source: 'Anatomy:Standard', description: 'Fat tissue' } },
  { id: 'urine_sample',        labels: ['Urine Sample'],                     type: 'MATERIAL',            metadata: { confidence: 0.96, source: 'Protocol:STANDARD', description: 'Non-invasive biofluid' } },
  { id: 'peripheral_blood',    labels: ['PBMCs'],                            type: 'MATERIAL',            metadata: { confidence: 0.94, source: 'Protocol:PMC6821', description: 'Peripheral blood mononuclear cells' } },
];

export const MOCK_TRIPLES = [
  { id: 't1',  subject: 'type2_diabetes',      relation: 'IS_ASSOCIATED_WITH', object: 'insulin_resistance',  confidence: 0.97 },
  { id: 't2',  subject: 'type2_diabetes',      relation: 'HAS_BIOMARKER',      object: 'hba1c',               confidence: 0.98 },
  { id: 't3',  subject: 'type2_diabetes',      relation: 'HAS_BIOMARKER',      object: 'fasting_glucose',     confidence: 0.97 },
  { id: 't4',  subject: 'type2_diabetes',      relation: 'CHARACTERIZED_BY',   object: 'metabolic_syndrome',  confidence: 0.88 },
  { id: 't5',  subject: 'insulin_resistance',  relation: 'MEASURED_BY',        object: 'homa_ir',             confidence: 0.93 },
  { id: 't6',  subject: 'insulin_resistance',  relation: 'INCREASES_LEVEL_OF', object: 'il6',                 confidence: 0.85 },
  { id: 't7',  subject: 'insulin_resistance',  relation: 'DECREASES_LEVEL_OF', object: 'adiponectin',         confidence: 0.87 },
  { id: 't8',  subject: 'metabolic_syndrome',  relation: 'INCLUDES_CRITERION', object: 'bmi',                 confidence: 0.92 },
  { id: 't9',  subject: 'metabolic_syndrome',  relation: 'INCLUDES_CRITERION', object: 'waist_circumference', confidence: 0.91 },
  { id: 't10', subject: 'metabolic_syndrome',  relation: 'CO_OCCURS_WITH',     object: 'nafld',               confidence: 0.86 },
  { id: 't11', subject: 'nafld',               relation: 'IS_ASSOCIATED_WITH', object: 'obesity',             confidence: 0.89 },
  { id: 't12', subject: 'obesity',             relation: 'MEASURED_BY',        object: 'bmi',                 confidence: 0.99 },
  { id: 't13', subject: 'obesity',             relation: 'MEASURED_BY',        object: 'waist_circumference', confidence: 0.95 },
  { id: 't14', subject: 'hba1c',              relation: 'QUANTIFIED_BY',      object: 'elisa',               confidence: 0.96 },
  { id: 't15', subject: 'hba1c',              relation: 'MEASURED_IN',        object: 'blood_plasma',        confidence: 0.97 },
  { id: 't16', subject: 'fasting_glucose',    relation: 'MEASURED_IN',        object: 'blood_plasma',        confidence: 0.98 },
  { id: 't17', subject: 'adiponectin',        relation: 'EXPRESSED_IN',       object: 'adipose_tissue',      confidence: 0.93 },
  { id: 't18', subject: 'il6',                relation: 'MEASURED_IN',        object: 'blood_plasma',        confidence: 0.95 },
  { id: 't19', subject: 'il6',                relation: 'UPREGULATED_IN',     object: 'adipose_tissue',      confidence: 0.88 },
  { id: 't20', subject: 'crp',                relation: 'INDICATES',          object: 'metabolic_syndrome',  confidence: 0.87 },
  { id: 't21', subject: 'crp',                relation: 'QUANTIFIED_BY',      object: 'elisa',               confidence: 0.97 },
  { id: 't22', subject: 'insulin_level',      relation: 'QUANTIFIED_BY',      object: 'elisa',               confidence: 0.97 },
  { id: 't23', subject: 'homa_ir',            relation: 'DERIVED_FROM',       object: 'fasting_glucose',     confidence: 0.99 },
  { id: 't24', subject: 'homa_ir',            relation: 'DERIVED_FROM',       object: 'insulin_level',       confidence: 0.99 },
  { id: 't25', subject: 'gc_ms',              relation: 'APPLIED_TO',         object: 'urine_sample',        confidence: 0.90 },
  { id: 't26', subject: 'gc_ms',              relation: 'ENABLES',            object: 'biomarker_discovery', confidence: 0.88 },
  { id: 't27', subject: 'rna_seq',            relation: 'APPLIED_TO',         object: 'adipose_tissue',      confidence: 0.93 },
  { id: 't28', subject: 'rna_seq',            relation: 'ENABLES',            object: 'pathway_analysis',    confidence: 0.92 },
  { id: 't29', subject: 'gwas',               relation: 'IDENTIFIES',         object: 'type2_diabetes',      confidence: 0.89 },
  { id: 't30', subject: 'gwas',               relation: 'APPLIED_TO',         object: 'peripheral_blood',    confidence: 0.95 },
  { id: 't31', subject: 'risk_stratification', relation: 'USES',              object: 'hba1c',               confidence: 0.93 },
  { id: 't32', subject: 'risk_stratification', relation: 'USES',              object: 'bmi',                 confidence: 0.91 },
  { id: 't33', subject: 'pathway_analysis',   relation: 'INVESTIGATES',       object: 'insulin_resistance',  confidence: 0.87 },
];

export const MOCK_RELATION_TYPES = [...new Set(MOCK_TRIPLES.map(t => t.relation))];

export const MOCK_SAMPLE_QA = [
  {
    id: 'qa1',
    question: 'Which biomarkers are elevated in metabolic syndrome and how are they measured?',
    answer: 'Metabolic syndrome presents a distinct inflammatory and metabolic biomarker signature. **C-Reactive Protein (CRP)** is a systemic inflammation marker consistently elevated in MetS, quantified via ELISA from blood plasma. **Interleukin-6 (IL-6)**, upregulated in adipose tissue, drives hepatic CRP synthesis.\n\nInsulin resistance is assessed via the **HOMA-IR score**, derived from fasting blood glucose and serum insulin. Anthropometric criteria **BMI** and **waist circumference** constitute WHO diagnostic criteria for MetS.',
    referencedNodes: ['crp', 'il6', 'insulin_level', 'metabolic_syndrome', 'elisa', 'blood_plasma', 'homa_ir', 'fasting_glucose', 'bmi', 'waist_circumference', 'adipose_tissue'],
    referencedEdges: ['t20', 't21', 't18', 't19', 't23', 't24', 't8', 't9', 't22'],
    model: 'GPT-4o', timestamp: '09:14', nodeCount: 11,
  },
  {
    id: 'qa2',
    question: 'What methods are used to study type 2 diabetes biomarkers?',
    answer: '**GC-MS metabolomics** applied to urine samples enables untargeted biomarker discovery. **RNA sequencing** of adipose tissue reveals transcriptional regulation. **GWAS** using peripheral blood mononuclear cells identifies genetic variants. **ELISA** quantifies circulating markers from blood plasma.\n\nThese platforms are increasingly combined in multi-omics validation pipelines.',
    referencedNodes: ['gc_ms', 'rna_seq', 'gwas', 'elisa', 'urine_sample', 'adipose_tissue', 'peripheral_blood', 'blood_plasma', 'hba1c', 'type2_diabetes', 'biomarker_discovery'],
    referencedEdges: ['t25', 't26', 't27', 't28', 't29', 't30', 't14', 't15'],
    model: 'GPT-5', timestamp: '10:02', nodeCount: 11,
  },
];
