"""
Weaviate Schema Definition for Viren
This module defines the schema for the Weaviate vector database
"""

def get_schema():
    """
    Returns the Weaviate schema for Viren
    """
    schema = {
        "classes": [
            {
                "class": "TechnicalKnowledge",
                "description": "Technical knowledge and documentation",
                "vectorizer": "text2vec-transformers",
                "moduleConfig": {
                    "text2vec-transformers": {
                        "vectorizeClassName": False
                    }
                },
                "properties": [
                    {
                        "name": "title",
                        "dataType": ["text"],
                        "description": "Title of the technical document",
                        "moduleConfig": {
                            "text2vec-transformers": {
                                "skip": False,
                                "vectorizePropertyName": False
                            }
                        }
                    },
                    {
                        "name": "content",
                        "dataType": ["text"],
                        "description": "Content of the technical document",
                        "moduleConfig": {
                            "text2vec-transformers": {
                                "skip": False,
                                "vectorizePropertyName": False
                            }
                        }
                    },
                    {
                        "name": "category",
                        "dataType": ["text"],
                        "description": "Category of the technical document",
                        "moduleConfig": {
                            "text2vec-transformers": {
                                "skip": False,
                                "vectorizePropertyName": False
                            }
                        }
                    },
                    {
                        "name": "source",
                        "dataType": ["text"],
                        "description": "Source of the technical document",
                        "moduleConfig": {
                            "text2vec-transformers": {
                                "skip": True
                            }
                        }
                    },
                    {
                        "name": "timestamp",
                        "dataType": ["date"],
                        "description": "Timestamp when the document was added",
                        "moduleConfig": {
                            "text2vec-transformers": {
                                "skip": True
                            }
                        }
                    }
                ]
            },
            {
                "class": "ProblemSolvingConcept",
                "description": "Problem-solving concepts and approaches",
                "vectorizer": "text2vec-transformers",
                "moduleConfig": {
                    "text2vec-transformers": {
                        "vectorizeClassName": False
                    }
                },
                "properties": [
                    {
                        "name": "name",
                        "dataType": ["text"],
                        "description": "Name of the problem-solving concept",
                        "moduleConfig": {
                            "text2vec-transformers": {
                                "skip": False,
                                "vectorizePropertyName": False
                            }
                        }
                    },
                    {
                        "name": "description",
                        "dataType": ["text"],
                        "description": "Description of the problem-solving concept",
                        "moduleConfig": {
                            "text2vec-transformers": {
                                "skip": False,
                                "vectorizePropertyName": False
                            }
                        }
                    },
                    {
                        "name": "examples",
                        "dataType": ["text[]"],
                        "description": "Examples of the problem-solving concept",
                        "moduleConfig": {
                            "text2vec-transformers": {
                                "skip": False,
                                "vectorizePropertyName": False
                            }
                        }
                    },
                    {
                        "name": "category",
                        "dataType": ["text"],
                        "description": "Category of the problem-solving concept",
                        "moduleConfig": {
                            "text2vec-transformers": {
                                "skip": False,
                                "vectorizePropertyName": False
                            }
                        }
                    }
                ]
            },
            {
                "class": "TroubleshootingTool",
                "description": "Troubleshooting tools and utilities",
                "vectorizer": "text2vec-transformers",
                "moduleConfig": {
                    "text2vec-transformers": {
                        "vectorizeClassName": False
                    }
                },
                "properties": [
                    {
                        "name": "name",
                        "dataType": ["text"],
                        "description": "Name of the troubleshooting tool",
                        "moduleConfig": {
                            "text2vec-transformers": {
                                "skip": False,
                                "vectorizePropertyName": False
                            }
                        }
                    },
                    {
                        "name": "description",
                        "dataType": ["text"],
                        "description": "Description of the troubleshooting tool",
                        "moduleConfig": {
                            "text2vec-transformers": {
                                "skip": False,
                                "vectorizePropertyName": False
                            }
                        }
                    },
                    {
                        "name": "usage",
                        "dataType": ["text"],
                        "description": "Usage instructions for the troubleshooting tool",
                        "moduleConfig": {
                            "text2vec-transformers": {
                                "skip": False,
                                "vectorizePropertyName": False
                            }
                        }
                    },
                    {
                        "name": "category",
                        "dataType": ["text"],
                        "description": "Category of the troubleshooting tool",
                        "moduleConfig": {
                            "text2vec-transformers": {
                                "skip": True
                            }
                        }
                    }
                ]
            },
            {
                "class": "BinaryMemoryShard",
                "description": "Binary memory shards for Viren",
                "vectorizer": "text2vec-transformers",
                "moduleConfig": {
                    "text2vec-transformers": {
                        "vectorizeClassName": False
                    }
                },
                "properties": [
                    {
                        "name": "shardId",
                        "dataType": ["string"],
                        "description": "Unique identifier for the memory shard",
                        "moduleConfig": {
                            "text2vec-transformers": {
                                "skip": True
                            }
                        }
                    },
                    {
                        "name": "content",
                        "dataType": ["text"],
                        "description": "Content of the memory shard",
                        "moduleConfig": {
                            "text2vec-transformers": {
                                "skip": False,
                                "vectorizePropertyName": False
                            }
                        }
                    },
                    {
                        "name": "metadata",
                        "dataType": ["text"],
                        "description": "Metadata for the memory shard",
                        "moduleConfig": {
                            "text2vec-transformers": {
                                "skip": True
                            }
                        }
                    },
                    {
                        "name": "timestamp",
                        "dataType": ["date"],
                        "description": "Timestamp when the shard was created",
                        "moduleConfig": {
                            "text2vec-transformers": {
                                "skip": True
                            }
                        }
                    },
                    {
                        "name": "priority",
                        "dataType": ["int"],
                        "description": "Priority level of the memory shard",
                        "moduleConfig": {
                            "text2vec-transformers": {
                                "skip": True
                            }
                        }
                    }
                ]
            }
        ]
    }
    
    return schema