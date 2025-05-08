import React, { useState } from 'react';
import {
  Drawer,
  Box,
  Typography,
  Divider,
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  IconButton,
  Chip,
  Button,
  Paper,
  Collapse,
  Alert,
  CircularProgress
} from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';
import AutoFixHighIcon from '@mui/icons-material/AutoFixHigh';
import CodeIcon from '@mui/icons-material/Code';
import DescriptionIcon from '@mui/icons-material/Description';
import ExtensionIcon from '@mui/icons-material/Extension';
import BugReportIcon from '@mui/icons-material/BugReport';
import SpeedIcon from '@mui/icons-material/Speed';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import ExpandLessIcon from '@mui/icons-material/ExpandLess';
import CheckCircleOutlineIcon from '@mui/icons-material/CheckCircleOutline';

import { 
  Recommendation, 
  RecommendationResponse, 
  RecommendationType,
  ApplyRecommendationResponse
} from './types';
import RecommendationPreview from './RecommendationPreview';
import { getRecommendationPreview, applyRecommendation, submitRecommendationFeedback } from './api';
import { useWorkflowContext } from '../workflow/WorkflowContext';

interface RecommendationsPanelProps {
  open: boolean;
  onClose: () => void;
  recommendations: RecommendationResponse;
  onRecommendationsUpdated: (recommendations: RecommendationResponse) => void;
}

const RecommendationsPanel: React.FC<RecommendationsPanelProps> = ({
  open,
  onClose,
  recommendations,
  onRecommendationsUpdated
}) => {
  const [selectedRecommendation, setSelectedRecommendation] = useState<Recommendation | null>(null);
  const [expandedRecommendations, setExpandedRecommendations] = useState<string[]>([]);
  const [previewingRecommendation, setPreviewingRecommendation] = useState<string | null>(null);
  const [applyingRecommendation, setApplyingRecommendation] = useState<string | null>(null);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  
  const { refreshWorkflow } = useWorkflowContext();

  const handleToggleExpand = (recommendationId: string) => {
    if (expandedRecommendations.includes(recommendationId)) {
      setExpandedRecommendations(expandedRecommendations.filter(id => id !== recommendationId));
    } else {
      setExpandedRecommendations([...expandedRecommendations, recommendationId]);
    }
  };

  const handleSelectRecommendation = (recommendation: Recommendation) => {
    setSelectedRecommendation(recommendation);
    
    // Expand this recommendation if not already expanded
    if (!expandedRecommendations.includes(recommendation.id)) {
      handleToggleExpand(recommendation.id);
    }
  };

  const handlePreviewRecommendation = async (recommendation: Recommendation) => {
    if (previewingRecommendation === recommendation.id) {
      setPreviewingRecommendation(null);
      return;
    }
    
    try {
      setPreviewingRecommendation(recommendation.id);
      // Actual preview loading happens in the RecommendationPreview component
    } catch (error) {
      console.error('Error previewing recommendation:', error);
      setErrorMessage('Failed to generate preview. Please try again.');
      setPreviewingRecommendation(null);
    }
  };

  const handleApplyRecommendation = async (recommendation: Recommendation, saveAsVersion: boolean = false) => {
    try {
      setApplyingRecommendation(recommendation.id);
      setErrorMessage(null);
      
      const result = await applyRecommendation(recommendation.id, {
        save_as_version: saveAsVersion,
        version_name: saveAsVersion ? `Applied "${recommendation.title}"` : undefined
      });
      
      // Update recommendation in the list
      const updatedRecommendations = {
        ...recommendations,
        recommendations: recommendations.recommendations.map(r => 
          r.id === recommendation.id 
            ? { ...r, applied: true, applied_at: new Date().toISOString() } 
            : r
        )
      };
      
      onRecommendationsUpdated(updatedRecommendations);
      
      // Submit feedback
      await submitRecommendationFeedback({
        recommendation_id: recommendation.id,
        useful: true,
        applied: true
      });
      
      // Show success message
      setSuccessMessage(
        saveAsVersion 
          ? `Successfully applied as new version "${result.new_version_id}"` 
          : 'Successfully applied recommendation!'
      );
      
      // Refresh the workflow to show changes
      await refresh

