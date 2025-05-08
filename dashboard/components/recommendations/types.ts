import { 
  RecommendationType, 
  RecommendationPriority, 
  RecommendationImpact 
} from '../../../backend/recommendations/models';

export interface Recommendation {
  id: string;
  workflow_id: string;
  type: RecommendationType;
  title: string;
  description: string;
  priority: RecommendationPriority;
  impact: RecommendationImpact;
  confidence: number;
  created_at: string;
  applied: boolean;
  applied_at?: string;
  module_suggestion?: ModuleSuggestion;
  prompt_enhancement?: PromptEnhancement;
  workflow_optimization?: WorkflowOptimization;
}

export interface ModuleSuggestion {
  module_id: string;
  module_name: string;
  module_type: string;
  description: string;
  insertion_point?: string;
  compatibility_score: number;
  configuration: Record<string, any>;
  reasoning: string;
}

export interface PromptEnhancement {
  node_id: string;
  original_prompt: string;
  enhanced_prompt: string;
  improvements: string[];
  expected_benefits: string[];
  reasoning: string;
  before_after_comparison?: Record<string, any>;
}

export interface WorkflowOptimization {
  optimization_type: string;
  affected_nodes: string[];
  description: string;
  expected_benefits: string[];
  implementation_complexity: string;
  before_diagram?: string;
  after_diagram?: string;
  reasoning: string;
}

export interface RecommendationResponse {
  workflow_id: string;
  recommendations: Recommendation[];
  analysis_summary: string;
  timestamp: string;
}

export interface RecommendationPreviewResponse {
  recommendation_id: string;
  workflow_id: string;
  before: Record<string, any>;
  after: Record<string, any>;
  changes_summary: string;
  can_apply: boolean;
  potential_issues?: string[];
}

export interface RecommendationFeedback {
  recommendation_id: string;
  useful: boolean;
  applied: boolean;
  rating?: number;
  comments?: string;
}

export interface ApplyRecommendationRequest {
  customizations?: Record<string, any>;
  save_as_version?: boolean;
  version_name?: string;
}

export interface ApplyRecommendationResponse {
  workflow_id: string;
  recommendation_id: string;
  applied: boolean;
  new_version_id?: string;
  changes: string[];
  timestamp: string;
}

