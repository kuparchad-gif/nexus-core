import React, { useState, useEffect } from "react";
import { Alert, Box, CircularProgress } from "@mui/material";

const MAX_SUBMISSIONS_PER_WEEK = 10;

function SubmissionLimitChecker({ user, children }) {
  const [loading, setLoading] = useState(true);
  const [reachedLimit, setReachedLimit] = useState(false);
  const [error, setError] = useState(false);

  useEffect(() => {
    const checkSubmissionLimit = async () => {
      if (!user?.username) {
        setLoading(false);
        return;
      }

      try {
        const response = await fetch(
          `/api/models/organization/${user.username}/submissions?days=7`
        );
        if (!response.ok) {
          throw new Error("Failed to fetch submission data");
        }

        const submissions = await response.json();
        console.log(`Recent submissions for ${user.username}:`, submissions);
        setReachedLimit(submissions.length >= MAX_SUBMISSIONS_PER_WEEK);
        setError(false);
      } catch (error) {
        console.error("Error checking submission limit:", error);
        setError(true);
      } finally {
        setLoading(false);
      }
    };

    checkSubmissionLimit();
  }, [user?.username]);

  if (loading) {
    return (
      <Box sx={{ display: "flex", justifyContent: "center", py: 4 }}>
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Alert
        severity="error"
        sx={{
          mb: 3,
          "& .MuiAlert-message": {
            fontSize: "1rem",
          },
        }}
      >
        Unable to verify submission limits. Please try again in a few minutes.
      </Alert>
    );
  }

  if (reachedLimit) {
    return (
      <Alert
        severity="warning"
        sx={{
          mb: 3,
          "& .MuiAlert-message": {
            fontSize: "1rem",
          },
        }}
      >
        For fairness reasons, you cannot submit more than{" "}
        {MAX_SUBMISSIONS_PER_WEEK} models per week. Please try again later.
      </Alert>
    );
  }

  return children;
}

export default SubmissionLimitChecker;
