"""
Multimodal Tree Decoder (MTD) for Table of Contents Extraction

Full implementation based on the paper:
"Multimodal Tree Decoder for Table of Contents Extraction in Document Images"
by Pengfei Hu, Zhenrong Zhang, Jianshu Zhang, Jun Du, Jiajia Wu

This module implements the complete MTD pipeline:
- Section III.A: Formalization
- Section III.B: Encoder (Vision + Text + Layout modules with Gated Unit)
- Section III.C: Classifier (BiGRU + Softmax)
- Section III.D: Decoder (Transformer + Attention + Relationship Prediction)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# CUDA Device Detection
# ============================================================================

def get_device():
    """
    Detect and return the best available device (CUDA GPU or CPU).
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"✓ CUDA available! Using GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"  CUDA version: {torch.version.cuda}")
        logger.info(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        return device
    else:
        device = torch.device('cpu')
        logger.info("⚠ CUDA not available. Using CPU (slower)")
        return device

# Global device - initialized once
DEVICE = get_device()


# ============================================================================
# Section III.A: Formalization - Data Structures
# ============================================================================

@dataclass
class Entity:
    """
    Represents a text entity (line) in the document.
    As defined in Section II (Dataset) and used in Section III.A (Formalization)
    """
    content: str                    # Textual content
    position: Tuple[float, float, float, float]  # (x0, y0, x1, y1)
    is_heading: bool = False        # Classification result
    heading_id: Optional[str] = None  # Hierarchical position (e.g., "2.2.1")
    index: int = 0                  # Position in document


@dataclass
class HeadingRelationship:
    """
    Represents relationship between two heading entities.
    As defined in Section III.A (Formalization)
    """
    from_entity_idx: int
    reference_entity_idx: int       # The entity immediately preceding
    relationship_type: str          # 'parent', 'sibling', or 'identity'
    confidence: float


# ============================================================================
# Section III.B: Encoder - Multimodal Feature Extraction
# ============================================================================

class VisionModule(nn.Module):
    """
    Vision Module using ResNet-34 + FPN
    As described in Section III.B.1 (Vision Module)

    "The vision module takes document images as input. It uses FPN to aggregate
    feature maps from ResNet-34, then pools a fixed-size feature map with RoIAlign
    for each entity."
    """

    def __init__(self, feature_dim=128):
        super().__init__()
        self.feature_dim = feature_dim

        # Use torchvision's ResNet-34 pretrained on ImageNet
        # In the paper, they pretrain on 1000 scientific documents for text detection
        # We'll use ImageNet pretrained weights as a reasonable approximation
        try:
            from torchvision.models import resnet34, ResNet34_Weights
            from torchvision.ops import RoIAlign

            # Load pretrained ResNet-34
            self.backbone = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)

            # Remove final FC layer and avgpool
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])

            # Simple FPN-like structure (paper uses full FPN)
            # We combine features from different layers
            self.fpn_conv = nn.Conv2d(512, feature_dim, kernel_size=1)

            # RoIAlign for extracting fixed-size features per entity
            # Paper uses 3x3 output size
            self.roi_align = RoIAlign(output_size=(3, 3), spatial_scale=1.0, sampling_ratio=2)

            # Flatten 3x3 features to feature_dim
            self.flatten_proj = nn.Linear(feature_dim * 3 * 3, feature_dim)

        except ImportError:
            logger.warning("torchvision not available, using fallback vision module")
            self.backbone = None
            self.fallback_proj = nn.Linear(4, feature_dim)  # Just use bbox coords

    def forward(self, image: torch.Tensor, bboxes: List[Tuple[float, float, float, float]]) -> torch.Tensor:
        """
        Extract visual features for each entity bounding box.

        Args:
            image: Document image tensor [C, H, W]
            bboxes: List of entity bounding boxes (x0, y0, x1, y1)

        Returns:
            Visual features tensor [N, feature_dim] where N is number of entities
        """
        if self.backbone is None:
            # Fallback: just use bbox coordinates as features
            bbox_tensor = torch.tensor(bboxes, dtype=torch.float32, device=DEVICE)
            return self.fallback_proj(bbox_tensor)

        # Move image to device
        image = image.to(DEVICE)

        # Extract features from ResNet-34 backbone
        with torch.no_grad():  # Paper freezes these during training
            features = self.backbone(image.unsqueeze(0))  # [1, 512, H', W']

        # Apply FPN conv
        features = self.fpn_conv(features)  # [1, feature_dim, H', W']

        # Extract features for each bbox using RoIAlign
        entity_features = []
        _, _, feat_h, feat_w = features.shape
        img_h, img_w = image.shape[1:]

        for bbox in bboxes:
            x0, y0, x1, y1 = bbox
            # Normalize bbox to feature map coordinates
            x0_norm = x0 / img_w * feat_w
            y0_norm = y0 / img_h * feat_h
            x1_norm = x1 / img_w * feat_w
            y1_norm = y1 / img_h * feat_h

            # RoIAlign expects boxes in format [batch_idx, x0, y0, x1, y1]
            roi_box = torch.tensor([[0, x0_norm, y0_norm, x1_norm, y1_norm]], dtype=torch.float32, device=DEVICE)

            # Extract fixed-size feature
            roi_feat = self.roi_align(features, roi_box)  # [1, feature_dim, 3, 3]

            # Flatten and project
            roi_feat_flat = roi_feat.view(1, -1)  # [1, feature_dim * 9]
            entity_feat = self.flatten_proj(roi_feat_flat)  # [1, feature_dim]
            entity_features.append(entity_feat)

        return torch.cat(entity_features, dim=0)  # [N, feature_dim]


class TextModule(nn.Module):
    """
    Text Module using BERT
    As described in Section III.B.2 (Text Module)

    "The BERT is used to extract the textual features. To make the extracted
    semantic features more suitable for our network, two linear transformations
    with a RELU activation are added following the BERT."
    """

    def __init__(self, feature_dim=128):
        super().__init__()
        self.feature_dim = feature_dim

        try:
            from transformers import BertModel, BertTokenizer

            # Load pretrained BERT
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.bert = BertModel.from_pretrained('bert-base-uncased')

            # Paper: "two linear transformations with a RELU activation"
            # BERT base outputs 768-dim features
            hidden_dim = 256
            self.fc1 = nn.Linear(768, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, feature_dim)
            self.relu = nn.ReLU()

            # Freeze BERT parameters (paper does this to save memory)
            for param in self.bert.parameters():
                param.requires_grad = False

        except ImportError:
            logger.warning("transformers not available, using fallback text module")
            self.tokenizer = None
            self.bert = None
            self.fallback_proj = nn.Linear(1, feature_dim)  # Use text length as feature

    def forward(self, texts: List[str]) -> torch.Tensor:
        """
        Extract textual features for each entity.

        Args:
            texts: List of text content for each entity

        Returns:
            Text features tensor [N, feature_dim]
        """
        if self.bert is None:
            # Fallback: use text length as feature
            lengths = torch.tensor([[len(t)] for t in texts], dtype=torch.float32, device=DEVICE)
            return self.fallback_proj(lengths)

        # Tokenize texts
        encoded = self.tokenizer(texts, padding=True, truncation=True,
                                max_length=128, return_tensors='pt')

        # Move encoded inputs to device
        encoded = {k: v.to(DEVICE) for k, v in encoded.items()}

        # Extract BERT features
        with torch.no_grad():  # Paper freezes BERT
            outputs = self.bert(**encoded)
            # Use [CLS] token embedding as sentence representation
            bert_features = outputs.last_hidden_state[:, 0, :]  # [N, 768]

        # Apply two linear transformations with ReLU
        features = self.relu(self.fc1(bert_features))  # [N, hidden_dim]
        features = self.fc2(features)  # [N, feature_dim]

        return features


class LayoutModule(nn.Module):
    """
    Layout Module for position features
    As described in Section III.B.3 (Layout Module)

    The paper computes 8-dimensional layout features:
    (x_lt/W, y_lt/H, x_rb/W, y_rb/H, w/w̄, h/h̄, (y_lt - y_{t-1}_rb)/h̄, (y_{t+1}_lt - y_rb)/h̄)
    """

    def __init__(self):
        super().__init__()
        # Layout features are computed, not learned
        # 8-dimensional as described in the paper

    def forward(self, bboxes: List[Tuple[float, float, float, float]],
                page_width: float, page_height: float) -> torch.Tensor:
        """
        Compute layout features for each entity.

        Args:
            bboxes: List of bounding boxes (x0, y0, x1, y1)
            page_width: Width of the document page
            page_height: Height of the document page

        Returns:
            Layout features tensor [N, 8]
        """
        if not bboxes:
            return torch.zeros((0, 8), dtype=torch.float32)

        # Calculate average width and height for normalization
        widths = [x1 - x0 for x0, y0, x1, y1 in bboxes]
        heights = [y1 - y0 for x0, y0, x1, y1 in bboxes]
        avg_width = np.mean(widths)
        avg_height = np.mean(heights)

        layout_features = []
        for i, (x0, y0, x1, y1) in enumerate(bboxes):
            w = x1 - x0
            h = y1 - y0

            # Normalized coordinates
            x_lt_norm = x0 / page_width
            y_lt_norm = y0 / page_height
            x_rb_norm = x1 / page_width
            y_rb_norm = y1 / page_height

            # Relative size
            w_rel = w / avg_width if avg_width > 0 else 1.0
            h_rel = h / avg_height if avg_height > 0 else 1.0

            # Spacing to previous entity
            if i > 0:
                prev_y_rb = bboxes[i-1][3]
                spacing_above = (y0 - prev_y_rb) / avg_height if avg_height > 0 else 0.0
            else:
                spacing_above = 0.0

            # Spacing to next entity
            if i < len(bboxes) - 1:
                next_y_lt = bboxes[i+1][1]
                spacing_below = (next_y_lt - y1) / avg_height if avg_height > 0 else 0.0
            else:
                spacing_below = 0.0

            features = [
                x_lt_norm, y_lt_norm, x_rb_norm, y_rb_norm,
                w_rel, h_rel, spacing_above, spacing_below
            ]
            layout_features.append(features)

        return torch.tensor(layout_features, dtype=torch.float32, device=DEVICE)


class GatedFusionUnit(nn.Module):
    """
    Gated Unit for fusing multimodal features
    As described in Section III.B.4 (Gated Unit)

    The paper uses:
    z_t = σ(W_z · [f^v_t, f^s_t, f^p_t])
    f_t = z_t * f^v_t + (1 - z_t) * f^s_t + E_z * f^p_t
    """

    def __init__(self, feature_dim=128, layout_dim=8):
        super().__init__()
        self.feature_dim = feature_dim
        self.layout_dim = layout_dim

        # Gate weights
        self.W_z = nn.Linear(2 * feature_dim + layout_dim, feature_dim)
        self.E_z = nn.Linear(layout_dim, feature_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, visual_feat: torch.Tensor, text_feat: torch.Tensor,
                layout_feat: torch.Tensor) -> torch.Tensor:
        """
        Fuse visual, text, and layout features using gated mechanism.

        Args:
            visual_feat: [N, feature_dim]
            text_feat: [N, feature_dim]
            layout_feat: [N, layout_dim]

        Returns:
            Fused features [N, feature_dim]
        """
        # Concatenate all features for gate computation
        concat_feat = torch.cat([visual_feat, text_feat, layout_feat], dim=1)

        # Compute gate values
        z = self.sigmoid(self.W_z(concat_feat))  # [N, feature_dim]

        # Fuse features
        layout_contribution = self.E_z(layout_feat)  # [N, feature_dim]
        fused = z * visual_feat + (1 - z) * text_feat + layout_contribution

        return fused


class MTDEncoder(nn.Module):
    """
    Complete MTD Encoder combining Vision, Text, Layout modules and Gated Fusion.
    As described in Section III.B (Encoder)
    """

    def __init__(self, feature_dim=128):
        super().__init__()
        self.vision_module = VisionModule(feature_dim)
        self.text_module = TextModule(feature_dim)
        self.layout_module = LayoutModule()
        self.gated_fusion = GatedFusionUnit(feature_dim, layout_dim=8)

    def forward(self, image: torch.Tensor, entities: List[Entity],
                page_width: float, page_height: float) -> torch.Tensor:
        """
        Extract and fuse multimodal features for all entities.

        Args:
            image: Document image [C, H, W]
            entities: List of Entity objects
            page_width: Page width
            page_height: Page height

        Returns:
            Fused features [N, feature_dim]
        """
        # Extract bounding boxes and texts
        bboxes = [e.position for e in entities]
        texts = [e.content for e in entities]

        # Extract features from each modality
        visual_features = self.vision_module(image, bboxes)
        text_features = self.text_module(texts)
        layout_features = self.layout_module(bboxes, page_width, page_height)

        # Fuse features using gated unit
        fused_features = self.gated_fusion(visual_features, text_features, layout_features)

        return fused_features


# ============================================================================
# Section III.C: Classifier - Heading Entity Detection
# ============================================================================

class MTDClassifier(nn.Module):
    """
    MTD Classifier using BiGRU + Fully Connected Layer
    As described in Section III.C (Classifier)

    "Before classification, Bidirectional Gated Recurrent Unit (BiGRU) is used
    to capture global information. Then we apply a fully connected layer and a
    softmax activation to classify each entity."
    """

    def __init__(self, feature_dim=128, hidden_dim=128):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim

        # BiGRU for capturing global context
        self.bigru = nn.GRU(feature_dim, hidden_dim, bidirectional=True, batch_first=True)

        # Fully connected layer for classification (heading vs normal)
        # BiGRU outputs hidden_dim * 2 (bidirectional)
        self.fc = nn.Linear(hidden_dim * 2, 2)

        # Paper uses focal loss, but we'll use cross-entropy for simplicity
        # Focal loss can be added if needed

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Classify entities as heading or normal.

        Args:
            features: Encoded features [N, feature_dim]

        Returns:
            Tuple of (classification scores [N, 2], hidden states [N, hidden_dim*2])
        """
        # BiGRU expects [batch, seq, feature]
        features = features.unsqueeze(0)  # [1, N, feature_dim]

        # Apply BiGRU to capture global information
        hidden_states, _ = self.bigru(features)  # [1, N, hidden_dim*2]
        hidden_states = hidden_states.squeeze(0)  # [N, hidden_dim*2]

        # Apply FC + softmax for classification
        logits = self.fc(hidden_states)  # [N, 2]
        scores = F.softmax(logits, dim=1)  # [N, 2]

        return scores, hidden_states


# ============================================================================
# Section III.D: Decoder - Tree Structure Building
# ============================================================================

class MTDDecoder(nn.Module):
    """
    MTD Decoder for building tree structure with attention mechanism.
    As described in Section III.D (Decoder)

    Uses Transformer + GRU + Attention + FFN to predict relationships.
    """

    def __init__(self, feature_dim=128, num_transformer_layers=3):
        super().__init__()
        self.feature_dim = feature_dim

        # Transformer for capturing long-range dependencies
        # Paper uses 3 layers
        encoder_layer = nn.TransformerEncoderLayer(d_model=feature_dim, nhead=4,
                                                   dim_feedforward=feature_dim*4)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)

        # GRU for sequential decoding
        self.gru = nn.GRUCell(feature_dim, feature_dim)

        # Attention mechanism components
        self.W_h = nn.Linear(feature_dim, feature_dim)
        self.W_m = nn.Linear(feature_dim, feature_dim)
        self.W_d = nn.Linear(1, feature_dim)
        self.v = nn.Parameter(torch.randn(feature_dim))

        # FFN for relationship prediction (parent, sibling, identity)
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, 3)  # 3 relationship types
        )

    def forward(self, heading_features: torch.Tensor, num_headings: int) -> List[HeadingRelationship]:
        """
        Build tree structure by predicting relationships between headings.

        Args:
            heading_features: Features of detected heading entities [M, feature_dim]
            num_headings: Number of heading entities

        Returns:
            List of HeadingRelationship objects
        """
        if num_headings == 0 or heading_features.size(0) == 0:
            return []

        # Apply transformer to capture long-range dependencies
        # Transformer expects [seq_len, batch, feature_dim]
        heading_features_t = heading_features.unsqueeze(1)  # [M, 1, feature_dim]
        heading_features_t = heading_features_t.permute(0, 1, 2)
        M = self.transformer(heading_features_t)  # [M, 1, feature_dim]
        M = M.squeeze(1)  # [M, feature_dim]

        relationships = []

        # Initialize GRU state on device
        h = torch.zeros(1, self.feature_dim, device=DEVICE)
        c = torch.zeros(1, self.feature_dim, device=DEVICE)

        # Track past attention (for coverage mechanism) on device
        past_attention = torch.zeros(num_headings, 1, device=DEVICE)

        # Sequentially decode relationships
        for s in range(1, num_headings):  # Start from 1 (first heading has no reference)
            # Compute hidden state prediction
            h_hat = self.gru(c, h)  # [1, feature_dim]

            # Compute attention energies
            # Paper uses coverage mechanism with past attention
            d = self.W_d(past_attention)  # [M, feature_dim]

            energies = []
            for i in range(num_headings):
                m_i = M[i].unsqueeze(0)  # [1, feature_dim]
                d_i = d[i].unsqueeze(0)  # [1, feature_dim]

                # Compute energy
                energy = torch.tanh(self.W_h(h_hat) + self.W_m(m_i) + d_i)
                energy = torch.dot(self.v, energy.squeeze())
                energies.append(energy)

            energies = torch.stack(energies)  # [M]

            # Find reference entity (argmax of energies)
            ref_idx = torch.argmax(energies).item()

            # Update attention tracking
            past_attention[ref_idx] += 1

            # Compute context vector
            attention_weight = torch.zeros_like(energies)
            attention_weight[ref_idx] = 1.0
            c_s = torch.sum(attention_weight.unsqueeze(1) * M, dim=0, keepdim=True)  # [1, feature_dim]

            # Update hidden state
            h = self.gru(c_s, h_hat)

            # Predict relationship type using FFN
            concat = torch.cat([c_s, h], dim=1)  # [1, feature_dim*2]
            rel_logits = self.ffn(concat)  # [1, 3]
            rel_probs = F.softmax(rel_logits, dim=1)
            rel_type_idx = torch.argmax(rel_probs).item()

            # Map to relationship type
            rel_types = ['parent', 'sibling', 'identity']
            rel_type = rel_types[rel_type_idx]
            confidence = rel_probs[0, rel_type_idx].item()

            relationships.append(HeadingRelationship(
                from_entity_idx=s,
                reference_entity_idx=ref_idx,
                relationship_type=rel_type,
                confidence=confidence
            ))

            c = c_s

        return relationships


# ============================================================================
# Complete MTD Model
# ============================================================================

class MTDModel(nn.Module):
    """
    Complete Multimodal Tree Decoder model.
    Combines Encoder, Classifier, and Decoder.
    """

    def __init__(self, feature_dim=128):
        super().__init__()
        self.encoder = MTDEncoder(feature_dim)
        self.classifier = MTDClassifier(feature_dim)
        self.decoder = MTDDecoder(feature_dim)

        # Move entire model to device (GPU if available)
        self.to(DEVICE)
        logger.info(f"✓ MTD Model moved to {DEVICE}")

    def forward(self, image: torch.Tensor, entities: List[Entity],
                page_width: float, page_height: float) -> Tuple[List[Entity], List[HeadingRelationship]]:
        """
        Complete MTD forward pass.

        Args:
            image: Document image [C, H, W]
            entities: List of Entity objects
            page_width: Page width
            page_height: Page height

        Returns:
            Tuple of (classified entities, relationships)
        """
        # 1. Encode: Extract multimodal features
        features = self.encoder(image, entities, page_width, page_height)

        # 2. Classify: Detect heading entities
        class_scores, _ = self.classifier(features)

        # Mark entities as headings based on classification
        heading_indices = []
        for i, entity in enumerate(entities):
            # class_scores[i, 1] is probability of being a heading
            if class_scores[i, 1] > 0.5:
                entity.is_heading = True
                heading_indices.append(i)

        # 3. Decode: Build tree structure for heading entities
        if len(heading_indices) > 0:
            heading_features = features[heading_indices]
            relationships = self.decoder(heading_features, len(heading_indices))
        else:
            relationships = []

        return entities, relationships


# ============================================================================
# High-Level TOC Detection Function
# ============================================================================

def detect_toc_with_mtd(image_path: str, min_headings: int = 4) -> Tuple[bool, float, Dict]:
    """
    Detect if a page is a Table of Contents using full MTD implementation.

    Args:
        image_path: Path to document image
        min_headings: Minimum number of headings required for TOC

    Returns:
        Tuple of (is_toc, confidence, metadata)
    """
    try:
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            return False, 0.0, {'error': 'Failed to load image'}

        img_h, img_w = img.shape[:2]

        # Convert to tensor [C, H, W]
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        # Extract entities using OCR (doctr with full text recognition)
        entities = extract_entities_with_ocr(image_path)

        if len(entities) < min_headings:
            return False, 0.0, {
                'reason': f'Too few entities: {len(entities)} < {min_headings}',
                'num_entities': len(entities)
            }

        # Initialize MTD model
        model = MTDModel(feature_dim=128)
        model.eval()

        # Run MTD inference
        with torch.no_grad():
            classified_entities, relationships = model(img_tensor, entities, img_w, img_h)

        # Count detected headings
        headings = [e for e in classified_entities if e.is_heading]
        num_headings = len(headings)

        # Calculate confidence based on multiple factors
        if num_headings < min_headings:
            is_toc = False
            confidence = 0.0
        else:
            # Check TOC-specific patterns in headings
            ends_with_numbers = sum(1 for h in headings if _ends_with_page_number(h.content))
            number_ratio = ends_with_numbers / num_headings if num_headings > 0 else 0

            # Check right alignment
            x_rights = [h.position[2] for h in headings]
            alignment_std = np.std(x_rights) / np.mean(x_rights) if x_rights else 1.0

            # Combine factors
            confidence = 0.0
            if number_ratio > 0.7:  # Most headings end with numbers
                confidence += 0.5
            if alignment_std < 0.1:  # Good right alignment
                confidence += 0.3
            if len(relationships) > 0:  # Has hierarchical structure
                confidence += 0.2

            is_toc = confidence >= 0.5

        metadata = {
            'num_entities': len(entities),
            'num_headings': num_headings,
            'num_relationships': len(relationships),
            'confidence': confidence
        }

        return is_toc, confidence, metadata

    except Exception as e:
        logger.error(f"MTD detection failed: {e}")
        return False, 0.0, {'error': str(e)}


def extract_entities_with_ocr(image_path: str) -> List[Entity]:
    """
    Extract entities (text lines) from document using doctr with full OCR.

    Args:
        image_path: Path to document image

    Returns:
        List of Entity objects with text content and bounding boxes
    """
    try:
        from doctr.io import DocumentFile
        from doctr.models import ocr_predictor

        # Use full OCR model (not just detection)
        model = ocr_predictor(pretrained=True)

        # Load document
        doc = DocumentFile.from_images(image_path)

        # Run OCR
        result = model(doc)

        # Extract entities from result
        entities = []
        entity_idx = 0

        # Navigate doctr result structure
        if hasattr(result, 'export'):
            doc_dict = result.export()
            for page in doc_dict.get('pages', []):
                page_h = page.get('dimensions', (0, 1))[1]
                page_w = page.get('dimensions', (1, 0))[0]

                for block in page.get('blocks', []):
                    for line in block.get('lines', []):
                        # Get line text
                        words = line.get('words', [])
                        text = ' '.join(w.get('value', '') for w in words)

                        if not text.strip():
                            continue

                        # Get line geometry (bounding box)
                        geometry = line.get('geometry', [[0, 0], [1, 1]])
                        if len(geometry) >= 2:
                            x0, y0 = geometry[0]
                            x1, y1 = geometry[1]

                            # Convert normalized coords to absolute
                            x0_abs = x0 * page_w
                            y0_abs = y0 * page_h
                            x1_abs = x1 * page_w
                            y1_abs = y1 * page_h

                            entity = Entity(
                                content=text,
                                position=(x0_abs, y0_abs, x1_abs, y1_abs),
                                index=entity_idx
                            )
                            entities.append(entity)
                            entity_idx += 1

        return entities

    except Exception as e:
        logger.error(f"OCR extraction failed: {e}")
        return []


def _ends_with_page_number(text: str) -> bool:
    """Check if text ends with a page number (arabic or roman numeral)."""
    import re
    words = text.strip().split()
    if not words:
        return False

    last_word = words[-1].replace('.', '').replace(',', '').strip()

    # Check arabic numerals
    if last_word.isdigit():
        return True

    # Check roman numerals
    if re.match(r'^[ivxlcdmIVXLCDM]+$', last_word):
        return True

    return False


# ============================================================================
# Testing and Debugging
# ============================================================================

if __name__ == "__main__":
    # Test the MTD model
    print("Testing MTD model components...")

    # Test encoder
    encoder = MTDEncoder(feature_dim=128)
    print(f"✓ MTD Encoder initialized")

    # Test classifier
    classifier = MTDClassifier(feature_dim=128)
    print(f"✓ MTD Classifier initialized")

    # Test decoder
    decoder = MTDDecoder(feature_dim=128)
    print(f"✓ MTD Decoder initialized")

    # Test complete model
    model = MTDModel(feature_dim=128)
    print(f"✓ Complete MTD Model initialized")

    print("\nAll MTD components initialized successfully!")
