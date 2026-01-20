# Test Metatron shard merging
crdt1 = NexusCRDT(shard=0, soul_id="test")
crdt2 = NexusCRDT(shard=0, soul_id="test")

crdt1.increment()  # +1 (Fib: 1)
crdt2.increment()  # +1 (Fib: 1) 
crdt1.increment()  # +1 (Fib: 2)

merged = crdt1.merge(crdt2)
print(f"Merged value: {merged.value}")  # Should be 4 (1+1+2)
print(f"Merged increments: {merged.incs}")  # Should be [1, 1, 2]