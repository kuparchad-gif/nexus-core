import { useMemo, useRef, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useSearchParams } from "react-router-dom";
import { useLeaderboard } from "../context/LeaderboardContext";
import { useDataProcessing } from "../components/Table/hooks/useDataProcessing";

export const useLeaderboardData = () => {
  const queryClient = useQueryClient();
  const [searchParams] = useSearchParams();
  const isInitialLoadRef = useRef(true);

  const { data, isLoading, error } = useQuery({
    queryKey: ["leaderboard"],
    queryFn: async () => {
      console.log("🔄 Starting API fetch attempt...");
      try {
        console.log("🌐 Fetching from API...");
        const response = await fetch("/api/leaderboard/formatted");
        console.log("📡 API Response status:", response.status);

        if (!response.ok) {
          const errorText = await response.text();
          console.error("🚨 API Error:", {
            status: response.status,
            statusText: response.statusText,
            body: errorText,
          });
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        const newData = await response.json();
        console.log("📥 Received data size:", JSON.stringify(newData).length);
        return newData;
      } catch (error) {
        console.error("🔥 Detailed error:", {
          name: error.name,
          message: error.message,
          stack: error.stack,
        });
        throw error;
      }
    },
    refetchOnWindowFocus: false,
    enabled: isInitialLoadRef.current || !!searchParams.toString(),
  });

  useMemo(() => {
    if (data && isInitialLoadRef.current) {
      console.log("🎯 Initial load complete");
      isInitialLoadRef.current = false;
    }
  }, [data]);

  return {
    data,
    isLoading,
    error,
    refetch: () => queryClient.invalidateQueries(["leaderboard"]),
  };
};

export const useLeaderboardProcessing = () => {
  const { state, actions } = useLeaderboard();
  const [sorting, setSorting] = useState([
    { id: "model.average_score", desc: true },
  ]);

  const memoizedData = useMemo(() => state.models, [state.models]);
  const memoizedFilters = useMemo(
    () => ({
      search: state.filters.search,
      precisions: state.filters.precisions,
      types: state.filters.types,
      paramsRange: state.filters.paramsRange,
      booleanFilters: state.filters.booleanFilters,
      isOfficialProviderActive: state.filters.isOfficialProviderActive,
    }),
    [
      state.filters.search,
      state.filters.precisions,
      state.filters.types,
      state.filters.paramsRange,
      state.filters.booleanFilters,
      state.filters.isOfficialProviderActive,
    ]
  );

  const {
    table,
    minAverage,
    maxAverage,
    getColorForValue,
    processedData,
    filteredData,
    columns,
    columnVisibility,
  } = useDataProcessing(
    memoizedData,
    memoizedFilters.search,
    memoizedFilters.precisions,
    memoizedFilters.types,
    memoizedFilters.paramsRange,
    memoizedFilters.booleanFilters,
    sorting,
    state.display.rankingMode,
    state.display.averageMode,
    state.display.visibleColumns,
    state.display.scoreDisplay,
    state.pinnedModels,
    actions.togglePinnedModel,
    setSorting,
    memoizedFilters.isOfficialProviderActive
  );

  return {
    table,
    minAverage,
    maxAverage,
    getColorForValue,
    processedData,
    filteredData,
    columns,
    columnVisibility,
    loading: state.loading,
    error: state.error,
  };
};
