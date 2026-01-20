# Loki / LogQL Quick Queries

## Raw stream
```
{app="viren"}
```

## Only allows / only denies
```
{app="viren"} |= "decision\": \"allow"
{app="viren"} |= "decision\": \"deny"
```

## Count over time (rate panels)
```
sum(count_over_time({app="viren"}[5m]))
sum(count_over_time({app="viren"} |= "decision\": \"allow"[5m]))
sum(count_over_time({app="viren"} |= "decision\": \"deny"[5m]))
```

## Parse JSON and filter
```
{app="viren"} | json | decision="allow"
{app="viren"} | json | decision="deny" | reason!~"consent"
```

## Top actions
```
topk(5, sum by (action) (count_over_time({app="viren"} | json | action!="" [1h])))
```
