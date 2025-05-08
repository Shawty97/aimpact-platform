import React, { useState } from 'react';
import {
  Fab,
  Tooltip,
  Badge,
  CircularProgress,
  Box,
  useTheme
} from '@mui/material';
import AutoFixHighIcon from '@mui/icons-material/AutoFixHigh';
import { useWorkflowContext } from '../workflow/WorkflowContext';
import RecommendationsPanel from './RecommendationsPanel';
import { getRecommendations } from './api';
import { RecommendationResponse } from './types';

interface RecommendationsButtonProps {
  position?: 'bottom-right' | 'bottom-left' | 'top-right' | 'top-left';
  size?: 'small' | 'medium' | 'large';
}

const RecommendationsButton: React.FC<RecommendationsButtonProps> = ({
  position = 'bottom-right',
  size = 'medium'
}) => {
  const [loading, setLoading] = useState(false);
  const [panelOpen, setPanelOpen] = useState(false);
  const [recommendations, setRecommendations] = useState<RecommendationResponse | null>(null);
  const { currentWorkflow } = useWorkflowContext();
  const theme = useTheme();

  // Position styles
  const positionStyles = {
    'bottom-right': {
      position: 'fixed',
      bottom: 16,
      right: 16,
    },
    'bottom-left': {
      position: 'fixed',
      bottom: 16,
      left: 16,
    },
    'top-right': {
      position: 'fixed',
      top: 16,
      right: 16,
    },
    'top-left': {
      position: 'fixed',
      top: 16,
      left: 16,
    }
  };

  const handleClick = async () => {
    if (!currentWorkflow) {
      console.error('No workflow selected');
      return;
    }
    
    if (panelOpen) {
      // If panel is already open, just close it
      setPanelOpen(false);
      return;
    }
    
    try {
      setLoading(true);
      const data = await getRecommendations(currentWorkflow.id);
      setRecommendations(data);
      setPanelOpen(true);
    } catch (error) {
      console.error('Error fetching recommendations:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleClosePanel = () => {
    setPanelOpen(false);
  };

  const recommendationCount = recommendations?.recommendations.length || 0;

  return (
    <>
      <Box sx={{ ...positionStyles[position], zIndex: 1050 }}>
        <Tooltip title="Make this workflow better">
          <Badge 
            badgeContent={recommendationCount > 0 ? recommendationCount : undefined} 
            color="secondary"
            overlap="circular"
          >
            <Fab
              color="primary"
              size={size}
              onClick={handleClick}
              disabled={loading || !currentWorkflow}
              sx={{
                background: theme.palette.primary.main,
                '&:hover': {
                  background: theme.palette.primary.dark,
                }
              }}
            >
              {loading ? (
                <CircularProgress size={24} color="inherit" />
              ) : (
                <AutoFixHighIcon />
              )}
            </Fab>
          </Badge>
        </Tooltip>
      </Box>
      
      {panelOpen && recommendations && (
        <RecommendationsPanel
          open={panelOpen}
          onClose={handleClosePanel}
          recommendations={recommendations}
          onRecommendationsUpdated={(updatedRecommendations) => {
            setRecommendations(updatedRecommendations);
          }}
        />
      )}
    </>
  );
};

export default RecommendationsButton;

