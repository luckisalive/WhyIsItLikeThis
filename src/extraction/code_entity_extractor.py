"""
Code entity extractor using tree-sitter AST parsing with Python ast module fallback.
Extracts functions, classes, and configuration keys from files and diffs.
"""

import ast
import json
import logging
import os
import re
from typing import List

import yaml

try:
    from tree_sitter import Language, Parser
    import tree_sitter_python
    HAS_TREE_SITTER = True
except ImportError:
    HAS_TREE_SITTER = False

from src.models import CodeEntity, EntityType

logger = logging.getLogger(__name__)


class CodeEntityExtractor:
    """Uses tree-sitter (with ast fallback) to extract code entities from files."""

    def __init__(self):
        """Initialize tree-sitter parser if available."""
        self.parser = None
        if HAS_TREE_SITTER:
            try:
                # Support tree-sitter 0.22+ and older
                try:
                    py_lang = Language(tree_sitter_python.language())
                except TypeError:
                    py_lang = Language(tree_sitter_python.language(), "python")

                try:
                    self.parser = Parser(py_lang)
                except (TypeError, AttributeError):
                    self.parser = Parser()
                    if hasattr(self.parser, "set_language"):
                        self.parser.set_language(py_lang)
                    else:
                        self.parser.language = py_lang
            except Exception as e:
                logger.warning(f"Could not initialize tree-sitter parser: {e}. Falling back to standard ast.")
                self.parser = None
        else:
            logger.info("Tree-sitter package not installed. Using standard ast module for Python parsing.")

    def _detect_language(self, path: str) -> str:
        """Detect language from file extension."""
        ext = os.path.splitext(path)[1].lower()
        if ext == ".py":
            return "python"
        elif ext in [".yaml", ".yml"]:
            return "yaml"
        elif ext == ".json":
            return "json"
        elif ext == ".toml":
            return "toml"
        return "unknown"

    def extract_from_file(self, file_path: str, content: bytes) -> List[CodeEntity]:
        """Extract code entities from file contents."""
        lang = self._detect_language(file_path)
        entities: List[CodeEntity] = []

        if lang == "python":
            # Primary: tree-sitter
            if self.parser:
                try:
                    tree = self.parser.parse(content)
                    entities = self._extract_python_entities_treesitter(tree.root_node, content, file_path)
                except Exception as e:
                    logger.warning(f"Tree-sitter parse failed for {file_path}: {e}. Retrying with ast.")
                    entities = self._extract_python_entities_ast(content, file_path)
            else:
                entities = self._extract_python_entities_ast(content, file_path)

        elif lang in ["yaml", "yml"]:
            entities = self._extract_yaml_keys(content, file_path)
        elif lang == "json":
            entities = self._extract_json_keys(content, file_path)

        return entities

    def _extract_python_entities_treesitter(self, root_node, content: bytes, file_path: str) -> List[CodeEntity]:
        """Extract classes and functions using tree-sitter CST."""
        entities: List[CodeEntity] = []

        def traverse(node):
            if node.type == "function_definition":
                name_node = node.child_by_field_name("name")
                if name_node:
                    name = content[name_node.start_byte : name_node.end_byte].decode("utf-8", errors="ignore")
                    entities.append(
                        CodeEntity(
                            name=name,
                            path=file_path,
                            type=EntityType.FUNCTION,
                            start_line=node.start_point[0] + 1,
                            end_line=node.end_point[0] + 1,
                            language="python",
                        )
                    )
            elif node.type == "class_definition":
                name_node = node.child_by_field_name("name")
                if name_node:
                    name = content[name_node.start_byte : name_node.end_byte].decode("utf-8", errors="ignore")
                    entities.append(
                        CodeEntity(
                            name=name,
                            path=file_path,
                            type=EntityType.CLASS,
                            start_line=node.start_point[0] + 1,
                            end_line=node.end_point[0] + 1,
                            language="python",
                        )
                    )

            for child in node.children:
                traverse(child)

        traverse(root_node)
        return entities

    def _extract_python_entities_ast(self, content: bytes, file_path: str) -> List[CodeEntity]:
        """Fallback Python entity extraction using standard library ast module."""
        entities: List[CodeEntity] = []
        try:
            tree = ast.parse(content.decode("utf-8", errors="ignore"), filename=file_path)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    entities.append(
                        CodeEntity(
                            name=node.name,
                            path=file_path,
                            type=EntityType.FUNCTION,
                            start_line=getattr(node, "lineno", 1),
                            end_line=getattr(node, "end_lineno", getattr(node, "lineno", 1)),
                            language="python",
                        )
                    )
                elif isinstance(node, ast.ClassDef):
                    entities.append(
                        CodeEntity(
                            name=node.name,
                            path=file_path,
                            type=EntityType.CLASS,
                            start_line=getattr(node, "lineno", 1),
                            end_line=getattr(node, "end_lineno", getattr(node, "lineno", 1)),
                            language="python",
                        )
                    )
        except Exception as e:
            logger.debug(f"AST parsing skipped for {file_path}: {e}")
        return entities

    def _extract_yaml_keys(self, content: bytes, file_path: str) -> List[CodeEntity]:
        """Extract top-level keys from YAML configuration files."""
        entities: List[CodeEntity] = []
        try:
            data = yaml.safe_load(content)
            if isinstance(data, dict):
                for key in data.keys():
                    entities.append(
                        CodeEntity(
                            name=str(key),
                            path=file_path,
                            type=EntityType.CONFIG_KEY,
                            start_line=1,
                            end_line=1,
                            language="yaml",
                        )
                    )
        except Exception as e:
            logger.debug(f"Failed to parse YAML file {file_path}: {e}")
        return entities

    def _extract_json_keys(self, content: bytes, file_path: str) -> List[CodeEntity]:
        """Extract top-level keys from JSON configuration files."""
        entities: List[CodeEntity] = []
        try:
            data = json.loads(content.decode("utf-8", errors="ignore"))
            if isinstance(data, dict):
                for key in data.keys():
                    entities.append(
                        CodeEntity(
                            name=str(key),
                            path=file_path,
                            type=EntityType.CONFIG_KEY,
                            start_line=1,
                            end_line=1,
                            language="json",
                        )
                    )
        except Exception as e:
            logger.debug(f"Failed to parse JSON file {file_path}: {e}")
        return entities

    def extract_from_diff(self, diff_text: str, file_path: str, content: bytes) -> List[str]:
        """Parse git diff hunk headers to get changed line ranges and return modified entity names."""
        if self._detect_language(file_path) != "python":
            return []

        modified_lines = set()
        for line in diff_text.split("\n"):
            match = re.match(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@", line)
            if match:
                start_line = int(match.group(1))
                count = int(match.group(2)) if match.group(2) else 1
                for i in range(count):
                    modified_lines.add(start_line + i)

        entities = self.extract_from_file(file_path, content)
        modified_entities = []

        for entity in entities:
            if entity.start_line is not None and entity.end_line is not None:
                entity_lines = set(range(entity.start_line, entity.end_line + 1))
                if modified_lines.intersection(entity_lines):
                    modified_entities.append(entity.name)

        return modified_entities
