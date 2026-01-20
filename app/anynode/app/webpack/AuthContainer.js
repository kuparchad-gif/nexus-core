import React from "react";
import {
  Box,
  Typography,
  Button,
  Chip,
  Stack,
  Paper,
  CircularProgress,
  useTheme,
  useMediaQuery,
} from "@mui/material";
import HFLogo from "../Logo/HFLogo";
import { useAuth } from "../../hooks/useAuth";
import LogoutIcon from "@mui/icons-material/Logout";
import { useNavigate } from "react-router-dom";

function AuthContainer({ actionText = "DO_ACTION" }) {
  const { isAuthenticated, user, login, logout, loading } = useAuth();
  const navigate = useNavigate();
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down("sm"));

  const handleLogout = () => {
    if (isAuthenticated && logout) {
      logout();
      navigate("/", { replace: true });
      window.location.reload();
    }
  };

  if (loading) {
    return (
      <Paper
        elevation={0}
        sx={{
          p: 3,
          mb: 4,
          border: "1px solid",
          borderColor: "grey.300",
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          gap: 2,
        }}
      >
        <CircularProgress size={24} />
      </Paper>
    );
  }

  if (!isAuthenticated) {
    return (
      <Paper
        elevation={0}
        sx={{
          p: 3,
          mb: 4,
          border: "1px solid",
          borderColor: "grey.300",
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          gap: 2,
        }}
      >
        <Typography variant="h6" align="center">
          Login to {actionText}
        </Typography>
        <Typography
          variant="body2"
          color="text.secondary"
          align="center"
          sx={{
            px: isMobile ? 2 : 0,
          }}
        >
          You need to be logged in with your Hugging Face account to{" "}
          {actionText.toLowerCase()}
        </Typography>
        <Button
          variant="contained"
          onClick={login}
          startIcon={
            <Box
              sx={{
                width: 20,
                height: 20,
                display: "flex",
                alignItems: "center",
              }}
            >
              <HFLogo />
            </Box>
          }
          sx={{
            textTransform: "none",
            fontWeight: 600,
            py: 1,
            px: 2,
            width: isMobile ? "100%" : "auto",
          }}
        >
          Sign in with Hugging Face
        </Button>
      </Paper>
    );
  }

  return (
    <Paper
      elevation={0}
      sx={{ p: 2, border: "1px solid", borderColor: "grey.300", mb: 4 }}
    >
      <Stack
        direction={isMobile ? "column" : "row"}
        spacing={2}
        alignItems={isMobile ? "stretch" : "center"}
        justifyContent="space-between"
      >
        <Stack
          direction={isMobile ? "column" : "row"}
          spacing={1}
          alignItems={isMobile ? "stretch" : "center"}
          sx={{ width: "100%" }}
        >
          <Typography
            variant="body1"
            align={isMobile ? "center" : "left"}
            sx={{ mb: isMobile ? 1 : 0 }}
          >
            Connected as <strong>{user?.username}</strong>
          </Typography>
          <Chip
            label={`Ready to ${actionText}`}
            color="success"
            size="small"
            variant="outlined"
            sx={{
              width: isMobile ? "100%" : "auto",
              height: isMobile ? 32 : 24,
              "& .MuiChip-label": {
                px: isMobile ? 2 : 1,
              },
            }}
          />
        </Stack>
        <Button
          variant="contained"
          onClick={handleLogout}
          endIcon={<LogoutIcon />}
          color="primary"
          sx={{
            minWidth: 120,
            height: 36,
            textTransform: "none",
            fontSize: "0.9375rem",
            width: isMobile ? "100%" : "auto",
          }}
        >
          Logout
        </Button>
      </Stack>
    </Paper>
  );
}

export default AuthContainer;
