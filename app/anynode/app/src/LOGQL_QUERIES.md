# LogQL Quickies
```
{app="viren"}
{app="viren"} |= "decision\": \"allow"
{app="viren"} |= "decision\": \"deny"
topk(5, sum by (action) (count_over_time({app="viren"} | json | action!="" [1h])))
```
