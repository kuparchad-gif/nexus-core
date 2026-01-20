import React, { useState, useEffect } from "react";
import {
  Box,
  Typography,
  Paper,
  Button,
  Alert,
  List,
  ListItem,
  CircularProgress,
  Chip,
  Divider,
  IconButton,
  Stack,
  Link,
  useTheme,
  useMediaQuery,
} from "@mui/material";
import AccessTimeIcon from "@mui/icons-material/AccessTime";
import PersonIcon from "@mui/icons-material/Person";
import OpenInNewIcon from "@mui/icons-material/OpenInNew";
import HowToVoteIcon from "@mui/icons-material/HowToVote";
import { useAuth } from "../../hooks/useAuth";
import PageHeader from "../../components/shared/PageHeader";
import AuthContainer from "../../components/shared/AuthContainer";
import { alpha } from "@mui/material/styles";
import CheckIcon from "@mui/icons-material/Check";

const NoModelsToVote = () => (
  <Box
    sx={{
      display: "flex",
      flexDirection: "column",
      alignItems: "center",
      justifyContent: "center",
      py: 8,
      textAlign: "center",
    }}
  >
    <HowToVoteIcon
      sx={{
        fontSize: 100,
        color: "grey.300",
        mb: 3,
      }}
    />
    <Typography
      variant="h4"
      component="h2"
      sx={{
        fontWeight: "bold",
        color: "grey.700",
        mb: 2,
      }}
    >
      No Models to Vote
    </Typography>
    <Typography
      variant="body1"
      sx={{
        color: "grey.600",
        maxWidth: 450,
        mx: "auto",
      }}
    >
      There are currently no models waiting for votes.
      <br />
      Check back later!
    </Typography>
  </Box>
);

const LOCAL_STORAGE_KEY = "pending_votes";

function VoteModelPage() {
  const { isAuthenticated, user, loading: authLoading } = useAuth();
  const [pendingModels, setPendingModels] = useState([]);
  const [loadingModels, setLoadingModels] = useState(true);
  const [error, setError] = useState(null);
  const [userVotes, setUserVotes] = useState(new Set());
  const [loadingVotes, setLoadingVotes] = useState({});
  const [localVotes, setLocalVotes] = useState(new Set());
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down("sm"));

  // Create a unique identifier for a model
  const getModelUniqueId = (model) => {
    return `${model.name}_${model.precision}_${model.revision}`;
  };

  const formatWaitTime = (submissionTime) => {
    if (!submissionTime) return "N/A";

    const now = new Date();
    const submitted = new Date(submissionTime);
    const diffInHours = Math.floor((now - submitted) / (1000 * 60 * 60));

    // Less than 24 hours: show in hours
    if (diffInHours < 24) {
      return `${diffInHours}h`;
    }

    // Less than 7 days: show in days
    const diffInDays = Math.floor(diffInHours / 24);
    if (diffInDays < 7) {
      return `${diffInDays}d`;
    }

    // More than 7 days: show in weeks
    const diffInWeeks = Math.floor(diffInDays / 7);
    return `${diffInWeeks}w`;
  };

  const getConfigVotes = (votesData, model) => {
    // Créer l'identifiant unique du modèle
    const modelUniqueId = getModelUniqueId(model);

    // Compter les votes du serveur
    let serverVotes = 0;
    for (const [key, config] of Object.entries(votesData.votes_by_config)) {
      if (
        config.precision === model.precision &&
        config.revision === model.revision
      ) {
        serverVotes = config.count;
        break;
      }
    }

    // Ajouter les votes en attente du localStorage
    const pendingVote = localVotes.has(modelUniqueId) ? 1 : 0;

    return serverVotes + pendingVote;
  };

  const sortModels = (models) => {
    // Trier d'abord par nombre de votes décroissant, puis par soumission de l'utilisateur
    return [...models].sort((a, b) => {
      // Comparer d'abord le nombre de votes
      if (b.votes !== a.votes) {
        return b.votes - a.votes;
      }

      // Si l'utilisateur est connecté, mettre ses modèles en priorité
      if (user) {
        const aIsUserModel = a.submitter === user.username;
        const bIsUserModel = b.submitter === user.username;

        if (aIsUserModel && !bIsUserModel) return -1;
        if (!aIsUserModel && bIsUserModel) return 1;
      }

      // Si égalité, trier par date de soumission (le plus récent d'abord)
      return new Date(b.submission_time) - new Date(a.submission_time);
    });
  };

  // Add this function to handle localStorage
  const updateLocalVotes = (modelUniqueId, action = "add") => {
    const storedVotes = JSON.parse(
      localStorage.getItem(LOCAL_STORAGE_KEY) || "[]"
    );
    if (action === "add") {
      if (!storedVotes.includes(modelUniqueId)) {
        storedVotes.push(modelUniqueId);
      }
    } else {
      const index = storedVotes.indexOf(modelUniqueId);
      if (index > -1) {
        storedVotes.splice(index, 1);
      }
    }
    localStorage.setItem(LOCAL_STORAGE_KEY, JSON.stringify(storedVotes));
    setLocalVotes(new Set(storedVotes));
  };

  useEffect(() => {
    const fetchData = async () => {
      try {
        // Ne pas afficher le loading si on a déjà des données
        if (pendingModels.length === 0) {
          setLoadingModels(true);
        }
        setError(null);

        // Charger d'abord les votes en attente du localStorage
        const storedVotes = JSON.parse(
          localStorage.getItem(LOCAL_STORAGE_KEY) || "[]"
        );
        const localVotesSet = new Set(storedVotes);

        // Préparer toutes les requêtes en parallèle
        const [pendingModelsResponse, userVotesResponse] = await Promise.all([
          fetch("/api/models/pending"),
          isAuthenticated && user
            ? fetch(`/api/votes/user/${user.username}`)
            : Promise.resolve(null),
        ]);

        if (!pendingModelsResponse.ok) {
          throw new Error("Failed to fetch pending models");
        }

        const modelsData = await pendingModelsResponse.json();
        const votedModels = new Set();

        // Traiter les votes de l'utilisateur si connecté
        if (userVotesResponse && userVotesResponse.ok) {
          const votesData = await userVotesResponse.json();
          const userVotes = Array.isArray(votesData) ? votesData : [];

          userVotes.forEach((vote) => {
            const uniqueId = `${vote.model}_${vote.precision || "unknown"}_${
              vote.revision || "main"
            }`;
            votedModels.add(uniqueId);
            if (localVotesSet.has(uniqueId)) {
              localVotesSet.delete(uniqueId);
              updateLocalVotes(uniqueId, "remove");
            }
          });
        }

        // Préparer et exécuter toutes les requêtes de votes en une seule fois
        const modelVotesResponses = await Promise.all(
          modelsData.map((model) => {
            const [provider, modelName] = model.name.split("/");
            return fetch(`/api/votes/model/${provider}/${modelName}`)
              .then((response) =>
                response.ok
                  ? response.json()
                  : { total_votes: 0, votes_by_config: {} }
              )
              .catch(() => ({ total_votes: 0, votes_by_config: {} }));
          })
        );

        // Construire les modèles avec toutes les données
        const modelsWithVotes = modelsData.map((model, index) => {
          const votesData = modelVotesResponses[index];
          const modelUniqueId = getModelUniqueId(model);
          const isVotedByUser =
            votedModels.has(modelUniqueId) || localVotesSet.has(modelUniqueId);

          return {
            ...model,
            votes: getConfigVotes(
              {
                ...votesData,
                votes_by_config: votesData.votes_by_config || {},
              },
              model
            ),
            votes_by_config: votesData.votes_by_config || {},
            wait_time: formatWaitTime(model.submission_time),
            hasVoted: isVotedByUser,
          };
        });

        // Mettre à jour tous les états en une seule fois
        const sortedModels = sortModels(modelsWithVotes);

        // Batch updates
        const updates = () => {
          setPendingModels(sortedModels);
          setUserVotes(votedModels);
          setLocalVotes(localVotesSet);
          setLoadingModels(false);
        };

        updates();
      } catch (err) {
        console.error("Error fetching data:", err);
        setError(err.message);
        setLoadingModels(false);
      }
    };

    fetchData();
  }, [isAuthenticated, user]);

  // Modify the handleVote function
  const handleVote = async (model) => {
    if (!isAuthenticated) return;

    const modelUniqueId = getModelUniqueId(model);

    try {
      setError(null);
      setLoadingVotes((prev) => ({ ...prev, [modelUniqueId]: true }));

      // Add to localStorage immediately
      updateLocalVotes(modelUniqueId, "add");

      // Encode model name for URL
      const encodedModelName = encodeURIComponent(model.name);

      const response = await fetch(
        `/api/votes/${encodedModelName}?vote_type=up&user_id=${user.username}`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            precision: model.precision,
            revision: model.revision,
          }),
        }
      );

      if (!response.ok) {
        // If the request fails, remove from localStorage
        updateLocalVotes(modelUniqueId, "remove");
        throw new Error("Failed to submit vote");
      }

      // Refresh votes for this model with cache bypass
      const [provider, modelName] = model.name.split("/");
      const timestamp = Date.now();
      const votesResponse = await fetch(
        `/api/votes/model/${provider}/${modelName}?nocache=${timestamp}`
      );

      if (!votesResponse.ok) {
        throw new Error("Failed to fetch updated votes");
      }

      const votesData = await votesResponse.json();
      console.log(`Updated votes for ${model.name}:`, votesData); // Debug log

      // Update model and resort the list
      setPendingModels((models) => {
        const updatedModels = models.map((m) =>
          getModelUniqueId(m) === getModelUniqueId(model)
            ? {
                ...m,
                votes: getConfigVotes(votesData, m),
                votes_by_config: votesData.votes_by_config || {},
                hasVoted: true,
              }
            : m
        );
        const sortedModels = sortModels(updatedModels);
        console.log("Updated and sorted models:", sortedModels); // Debug log
        return sortedModels;
      });

      // Update user votes with unique ID
      setUserVotes((prev) => new Set([...prev, getModelUniqueId(model)]));
    } catch (err) {
      console.error("Error voting:", err);
      setError(err.message);
    } finally {
      // Clear loading state for this model
      setLoadingVotes((prev) => ({
        ...prev,
        [modelUniqueId]: false,
      }));
    }
  };

  // Modify the rendering logic to consider both server and local votes
  // Inside the map function where you render models
  const isVoted = (model) => {
    const modelUniqueId = getModelUniqueId(model);
    return userVotes.has(modelUniqueId) || localVotes.has(modelUniqueId);
  };

  if (authLoading || (loadingModels && pendingModels.length === 0)) {
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
    <Box
      sx={{
        width: "100%",
        maxWidth: 1200,
        margin: "0 auto",
        py: 4,
        px: 0,
      }}
    >
      <PageHeader
        title="Vote for the Next Models"
        subtitle={
          <>
            Help us <span style={{ fontWeight: 600 }}>prioritize</span> which
            models to evaluate next
          </>
        }
      />

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      {/* Auth Status */}
      {/* <Box sx={{ mb: 3 }}>
        {isAuthenticated ? (
          <Paper
            elevation={0}
            sx={{ p: 2, border: "1px solid", borderColor: "grey.300" }}
          >
            <Stack
              direction="row"
              spacing={2}
              alignItems="center"
              justifyContent="space-between"
            >
              <Stack direction="row" spacing={1} alignItems="center">
                <Typography variant="body1">
                  Connected as <strong>{user?.username}</strong>
                </Typography>
                <Chip
                  label="Ready to vote"
                  color="success"
                  size="small"
                  variant="outlined"
                />
              </Stack>
              <LogoutButton />
            </Stack>
          </Paper>
        ) : (
          <Paper
            elevation={0}
            sx={{
              p: 3,
              border: "1px solid",
              borderColor: "grey.300",
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              gap: 2,
            }}
          >
            <Typography variant="h6" align="center">
              Login to Vote
            </Typography>
            <Typography variant="body2" color="text.secondary" align="center">
              You need to be logged in with your Hugging Face account to vote
              for models
            </Typography>
            <AuthBlock />
          </Paper>
        )}
      </Box> */}
      <AuthContainer actionText="vote for models" />

      {/* Models List */}
      <Paper
        elevation={0}
        sx={{
          border: "1px solid",
          borderColor: "grey.300",
          borderRadius: 1,
          overflow: "hidden",
          minHeight: 400,
        }}
      >
        {/* Header - Always visible */}
        <Box
          sx={{
            px: 3,
            py: 2,
            borderBottom: "1px solid",
            borderColor: (theme) =>
              theme.palette.mode === "dark"
                ? alpha(theme.palette.divider, 0.1)
                : "grey.200",
            bgcolor: (theme) =>
              theme.palette.mode === "dark"
                ? alpha(theme.palette.background.paper, 0.5)
                : "grey.50",
          }}
        >
          <Typography
            variant="h6"
            sx={{ fontWeight: 600, color: "text.primary" }}
          >
            Models Pending Evaluation
          </Typography>
        </Box>

        {/* Table Header */}
        <Box
          sx={{
            px: 3,
            py: 1.5,
            borderBottom: "1px solid",
            borderColor: "divider",
            bgcolor: "background.paper",
            display: { xs: "none", sm: "grid" },
            gridTemplateColumns: "1fr 200px 160px",
            gap: 3,
            alignItems: "center",
          }}
        >
          <Box>
            <Typography variant="subtitle2" color="text.secondary">
              Model
            </Typography>
          </Box>
          <Box sx={{ textAlign: "right" }}>
            <Typography variant="subtitle2" color="text.secondary">
              Votes
            </Typography>
          </Box>
          <Box sx={{ textAlign: "right" }}>
            <Typography variant="subtitle2" color="text.secondary">
              Priority
            </Typography>
          </Box>
        </Box>

        {/* Content */}
        {loadingModels ? (
          <Box
            sx={{
              display: "flex",
              justifyContent: "center",
              alignItems: "center",
              height: "200px",
              width: "100%",
              bgcolor: "background.paper",
            }}
          >
            <CircularProgress />
          </Box>
        ) : pendingModels.length === 0 && !loadingModels ? (
          <NoModelsToVote />
        ) : (
          <List sx={{ p: 0, bgcolor: "background.paper" }}>
            {pendingModels.map((model, index) => {
              const isTopThree = index < 3;
              return (
                <React.Fragment key={getModelUniqueId(model)}>
                  {index > 0 && <Divider />}
                  <ListItem
                    sx={{
                      py: 2.5,
                      px: 3,
                      display: "grid",
                      gridTemplateColumns: { xs: "1fr", sm: "1fr 200px 160px" },
                      gap: { xs: 2, sm: 3 },
                      alignItems: "start",
                      position: "relative",
                      "&:hover": {
                        bgcolor: "action.hover",
                      },
                    }}
                  >
                    {/* Left side - Model info */}
                    <Box>
                      <Stack spacing={1}>
                        {/* Model name and link */}
                        <Stack
                          direction={{ xs: "column", sm: "row" }}
                          spacing={1}
                          alignItems={{ xs: "stretch", sm: "center" }}
                        >
                          <Stack
                            direction="row"
                            spacing={1}
                            alignItems="center"
                            sx={{ flexGrow: 1 }}
                          >
                            <Link
                              href={`https://huggingface.co/${model.name}`}
                              target="_blank"
                              rel="noopener noreferrer"
                              sx={{
                                textDecoration: "none",
                                color: "primary.main",
                                fontWeight: 500,
                                "&:hover": {
                                  textDecoration: "underline",
                                },
                                fontSize: { xs: "0.9rem", sm: "inherit" },
                                wordBreak: "break-word",
                              }}
                            >
                              {model.name}
                            </Link>
                            <IconButton
                              size="small"
                              href={`https://huggingface.co/${model.name}`}
                              target="_blank"
                              rel="noopener noreferrer"
                              sx={{
                                ml: 0.5,
                                p: 0.5,
                                color: "action.active",
                                "&:hover": {
                                  color: "primary.main",
                                },
                              }}
                            >
                              <OpenInNewIcon sx={{ fontSize: "1rem" }} />
                            </IconButton>
                          </Stack>
                          <Stack
                            direction="row"
                            spacing={1}
                            sx={{
                              width: { xs: "100%", sm: "auto" },
                              justifyContent: {
                                xs: "flex-start",
                                sm: "flex-end",
                              },
                              flexWrap: "wrap",
                              gap: 1,
                            }}
                          >
                            <Chip
                              label={model.precision}
                              size="small"
                              variant="outlined"
                              sx={{
                                borderColor: "grey.300",
                                bgcolor: "grey.50",
                                "& .MuiChip-label": {
                                  fontSize: "0.75rem",
                                  fontWeight: 600,
                                  color: "text.secondary",
                                },
                              }}
                            />
                            <Chip
                              label={`rev: ${model.revision.slice(0, 7)}`}
                              size="small"
                              variant="outlined"
                              sx={{
                                borderColor: "grey.300",
                                bgcolor: "grey.50",
                                "& .MuiChip-label": {
                                  fontSize: "0.75rem",
                                  fontWeight: 600,
                                  color: "text.secondary",
                                },
                              }}
                            />
                          </Stack>
                        </Stack>
                        {/* Metadata row */}
                        <Stack
                          direction={{ xs: "column", sm: "row" }}
                          spacing={{ xs: 1, sm: 2 }}
                          alignItems={{ xs: "flex-start", sm: "center" }}
                        >
                          <Stack
                            direction="row"
                            spacing={0.5}
                            alignItems="center"
                          >
                            <AccessTimeIcon
                              sx={{
                                fontSize: "0.875rem",
                                color: "text.secondary",
                              }}
                            />
                            <Typography variant="body2" color="text.secondary">
                              {model.wait_time}
                            </Typography>
                          </Stack>
                          <Stack
                            direction="row"
                            spacing={0.5}
                            alignItems="center"
                          >
                            <PersonIcon
                              sx={{
                                fontSize: "0.875rem",
                                color: "text.secondary",
                              }}
                            />
                            <Typography variant="body2" color="text.secondary">
                              {model.submitter}
                            </Typography>
                          </Stack>
                        </Stack>
                      </Stack>
                    </Box>

                    {/* Vote Column */}
                    <Box
                      sx={{
                        textAlign: { xs: "left", sm: "right" },
                        mt: { xs: 2, sm: 0 },
                      }}
                    >
                      <Stack
                        direction={{ xs: "row", sm: "row" }}
                        spacing={2.5}
                        justifyContent={{ xs: "space-between", sm: "flex-end" }}
                        alignItems="center"
                      >
                        <Stack
                          alignItems={{ xs: "flex-start", sm: "center" }}
                          sx={{
                            minWidth: { xs: "auto", sm: "90px" },
                          }}
                        >
                          <Typography
                            variant="h4"
                            component="div"
                            sx={{
                              fontWeight: 700,
                              lineHeight: 1,
                              fontSize: { xs: "1.75rem", sm: "2rem" },
                              display: "flex",
                              alignItems: "center",
                              justifyContent: "center",
                            }}
                          >
                            <Typography
                              component="span"
                              sx={{
                                fontSize: { xs: "1.25rem", sm: "1.5rem" },
                                fontWeight: 600,
                                color: "primary.main",
                                lineHeight: 1,
                                mr: 0.5,
                                mt: "-2px",
                              }}
                            >
                              +
                            </Typography>
                            <Typography
                              component="span"
                              sx={{
                                color:
                                  model.votes === 0
                                    ? "text.primary"
                                    : "primary.main",
                                fontWeight: 700,
                                lineHeight: 1,
                              }}
                            >
                              {model.votes > 999 ? "999" : model.votes}
                            </Typography>
                          </Typography>
                          <Typography
                            variant="caption"
                            sx={{
                              color: "text.secondary",
                              fontWeight: 500,
                              mt: 0.5,
                              textTransform: "uppercase",
                              letterSpacing: "0.05em",
                              fontSize: "0.75rem",
                            }}
                          >
                            votes
                          </Typography>
                        </Stack>
                        <Button
                          variant={isVoted(model) ? "contained" : "outlined"}
                          size={isMobile ? "medium" : "large"}
                          onClick={() => handleVote(model)}
                          disabled={
                            !isAuthenticated ||
                            isVoted(model) ||
                            loadingVotes[getModelUniqueId(model)]
                          }
                          color="primary"
                          sx={{
                            minWidth: { xs: "80px", sm: "100px" },
                            height: { xs: "36px", sm: "40px" },
                            textTransform: "none",
                            fontWeight: 600,
                            fontSize: { xs: "0.875rem", sm: "0.95rem" },
                            ...(isVoted(model)
                              ? {
                                  bgcolor: "primary.main",
                                  "&:hover": {
                                    bgcolor: "primary.dark",
                                  },
                                  "&.Mui-disabled": {
                                    bgcolor: "primary.main",
                                    color: "white",
                                    opacity: 0.7,
                                  },
                                }
                              : {
                                  borderWidth: 2,
                                  "&:hover": {
                                    borderWidth: 2,
                                  },
                                }),
                          }}
                        >
                          {loadingVotes[getModelUniqueId(model)] ? (
                            <CircularProgress size={20} color="inherit" />
                          ) : isVoted(model) ? (
                            <Stack
                              direction="row"
                              spacing={0.5}
                              alignItems="center"
                            >
                              <CheckIcon sx={{ fontSize: "1.2rem" }} />
                              <span>Voted</span>
                            </Stack>
                          ) : (
                            "Vote"
                          )}
                        </Button>
                      </Stack>
                    </Box>

                    {/* Priority Column */}
                    <Box
                      sx={{
                        textAlign: { xs: "left", sm: "right" },
                        mt: { xs: 2, sm: 0 },
                        display: { xs: "none", sm: "block" },
                      }}
                    >
                      <Chip
                        label={
                          <Stack
                            direction="row"
                            spacing={0.5}
                            alignItems="center"
                          >
                            {isTopThree && (
                              <Typography
                                variant="body2"
                                sx={{
                                  fontWeight: 600,
                                  color: isTopThree
                                    ? "primary.main"
                                    : "text.primary",
                                  letterSpacing: "0.02em",
                                }}
                              >
                                HIGH
                              </Typography>
                            )}
                            <Typography
                              variant="body2"
                              sx={{
                                fontWeight: 600,
                                color: isTopThree
                                  ? "primary.main"
                                  : "text.secondary",
                                letterSpacing: "0.02em",
                              }}
                            >
                              #{index + 1}
                            </Typography>
                          </Stack>
                        }
                        size="medium"
                        variant={isTopThree ? "filled" : "outlined"}
                        sx={{
                          height: 36,
                          minWidth: "100px",
                          bgcolor: isTopThree
                            ? (theme) => alpha(theme.palette.primary.main, 0.1)
                            : "transparent",
                          borderColor: isTopThree ? "primary.main" : "grey.300",
                          borderWidth: 2,
                          "& .MuiChip-label": {
                            px: 2,
                            fontSize: "0.95rem",
                          },
                        }}
                      />
                    </Box>
                  </ListItem>
                </React.Fragment>
              );
            })}
          </List>
        )}
      </Paper>
    </Box>
  );
}

export default VoteModelPage;
