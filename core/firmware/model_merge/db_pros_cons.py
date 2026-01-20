# Store everything, merge via queries
db = WeightDatabase()
db.import_model("model_a.safetensors", "model_a")
db.import_model("model_b.safetensors", "model_b") 

# Create fusion recipe
recipe_id = db.create_fusion_recipe(["model_a", "model_b"], "quality_weighted")

# Execute fusion
fused_model = db.execute_fusion(recipe_id)
# ✅ All weights remain accessible
# ✅ Can try multiple fusion strategies
# ✅ Historical tracking of fusion attempts