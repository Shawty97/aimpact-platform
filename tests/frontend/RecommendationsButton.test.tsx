import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import RecommendationsButton from '../../dashboard/components/recommendations/RecommendationsButton';
import { getRecommendations } from '../../dashboard/components/recommendations/api';
import { useWorkflowContext } from '../../dashboard/components/workflow/WorkflowContext';
import { RecommendationResponse } from '../../dashboard/components/recommendations/types';

// Mock the API
jest.mock('../../dashboard/components/recommendations/api', () => ({
  getRecommendations: jest.fn()
}));

// Mock the workflow context
jest.mock('../../dashboard/components/workflow/WorkflowContext', () => ({
  useWorkflowContext: jest.fn()
}));

// Mock the Recommendations Panel component
jest.mock('../../dashboard/components/recommendations/RecommendationsPanel', () => {
  return function DummyRecommendationsPanel(props: any) {
    return (
      <div data-testid="recommendations-panel">
        <div>Recommendations Panel</div>
        <button onClick={props.onClose}>Close</button>
      </div>
    );
  };
});

// Sample data
const mockWorkflow = {
  id: '123',
  name: 'Test Workflow',
  nodes: [],
  edges: []
};

const mockRecommendations: RecommendationResponse = {
  workflow_id: '123',
  recommendations: [
    {
      

