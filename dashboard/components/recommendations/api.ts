import axios from 'axios';
import { 
  Recommendation, 
  RecommendationResponse,
  RecommendationPreviewResponse,
  RecommendationFeedback,
  ApplyRecommendationRequest,
  ApplyRecommendationResponse
} from './types';

const API_BASE_URL = '/api/recommendations';

/**
 * Fetches recommendations for a workflow
 */
export const getRecommendations = async (
  workflowId: string, 
  options?: { 
    focus_areas?: string[],
    node_ids?: string[],
    max_suggestions?: number,
    min_confidence?: number,
    include_reasoning?: boolean
  }
): Promise<RecommendationResponse> => {
  const params = new URLSearchParams();
  
  if (options?.focus_areas?.length) {
    options.focus_areas.forEach(area => params.append('focus_areas', area));
  }
  
  if (options?.node_ids?.length) {
    options.node_ids.forEach(id => params.append('node_ids', id));
  }
  
  if (options?.max_suggestions) {
    params.append('max_suggestions', options.max_suggestions.toString());
  }
  
  if (options?.min_confidence) {
    params.append('min_confidence', options.min_confidence.toString());
  }
  
  if (options?.include_reasoning !== undefined) {
    params.append('include_reasoning', options.include_reasoning.toString());
  }
  
  const response = await axios.get(`${API_BASE_URL}/workflow/${workflowId}`, { params });
  return response.data;
};

/**
 * Fetches a preview of applying a recommendation
 */
export const getRecommendationPreview = async (
  recommendationId: string,
  customizations?: Record<string, any>
): Promise<RecommendationPreviewResponse> => {
  const params = new URLSearchParams();
  
  if (customizations) {
    params.append('customizations', JSON.stringify(customizations));
  }
  
  const response = await axios.get(`${API_BASE_URL}/preview/${recommendationId}`, { params });
  return response.data;
};

/**
 * Applies a recommendation to a workflow
 */
export const applyRecommendation = async (
  recommendationId: string,
  options: ApplyRecommendationRequest
): Promise<ApplyRecommendationResponse> => {
  const response = await axios.post(`${API_BASE_URL}/apply/${recommendationId}`, options);
  return response.data;
};

/**
 * Submits feedback on a recommendation
 */
export const submitRecommendationFeedback = async (
  feedback: RecommendationFeedback
): Promise<void> => {
  await axios.post(`${API_BASE_URL}/feedback`, feedback);
};

