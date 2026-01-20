import React from "react";
import { Box, CircularProgress } from "@mui/material";
import { useAuth } from "../../hooks/useAuth";
import PageHeader from "../../components/shared/PageHeader";
import EvaluationQueues from "./components/EvaluationQueues/EvaluationQueues";
import ModelSubmissionForm from "./components/ModelSubmissionForm/ModelSubmissionForm";
import SubmissionGuide from "./components/SubmissionGuide/SubmissionGuide";
import SubmissionLimitChecker from "./components/SubmissionLimitChecker/SubmissionLimitChecker";

function AddModelPage() {
  const { isAuthenticated, loading, user } = useAuth();

  if (loading) {
    return (
      <Box
        sx={{
          display: "flex",
          justifyContent: "center",
          alignItems: "center",
          height: "100vh",
        }}
      >
        <CircularProgress />
      </Box>
    );
  }

  return (
    <Box sx={{ width: "100%", maxWidth: 1200, margin: "0 auto", py: 4, px: 0 }}>
      <PageHeader
        title="Submit a Model for Evaluation"
        subtitle={
          <>
            Add <span style={{ fontWeight: 600 }}>your</span> model to the Open
            LLM Leaderboard
          </>
        }
      />

      <SubmissionGuide />

      <SubmissionLimitChecker user={user}>
        <ModelSubmissionForm user={user} isAuthenticated={isAuthenticated} />
      </SubmissionLimitChecker>

      <EvaluationQueues defaultExpanded={false} />
    </Box>
  );
}

export default AddModelPage;
