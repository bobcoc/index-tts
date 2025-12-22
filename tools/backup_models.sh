#!/bin/bash
# =============================================================================
# Model Backup and Restore Script for IndexTTS + LatentSync
# =============================================================================
#
# Usage:
#   Backup:  ./backup_models.sh backup [backup_dir]
#   Restore: ./backup_models.sh restore [backup_dir]
#   List:    ./backup_models.sh list [backup_dir]
#
# Default backup directory: ~/model_backups
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DEFAULT_BACKUP_DIR="$HOME/model_backups"

# Model paths to backup
INDEXTTS_CHECKPOINTS="$PROJECT_DIR/checkpoints"
LATENTSYNC_DIR="$HOME/LatentSync"
INSIGHTFACE_DIR="$HOME/.insightface/models"
HUGGINGFACE_CACHE="$HOME/.cache/huggingface/hub"

# Files/directories to backup from IndexTTS
INDEXTTS_FILES=(
    "config.yaml"
    "gpt.pth"
    "bpe.model"
    "wav2vec2bert_stats.pt"
    "s2mel.pth"
    "feat1.pt"
    "feat2.pt"
    "qwen0.6bemo4-merge"
    "glossary.yaml"
)

# LatentSync checkpoints
LATENTSYNC_FILES=(
    "checkpoints/latentsync_unet.pt"
    "checkpoints/whisper/tiny.pt"
)

# InsightFace models
INSIGHTFACE_MODELS=(
    "buffalo_l"
)

# HuggingFace models to backup (optional, large files)
HF_MODELS=(
    "models--nvidia--bigvgan_v2_22khz_80band_256x"
    "models--amphion--MaskGCT"
    "models--funasr--campplus"
)

print_header() {
    echo -e "${BLUE}==========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}==========================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

get_size() {
    if [ -e "$1" ]; then
        du -sh "$1" 2>/dev/null | cut -f1
    else
        echo "N/A"
    fi
}

backup_models() {
    local backup_dir="${1:-$DEFAULT_BACKUP_DIR}"
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local backup_path="$backup_dir/backup_$timestamp"
    
    print_header "Model Backup"
    echo "Backup directory: $backup_path"
    echo ""
    
    mkdir -p "$backup_path"
    
    # Backup IndexTTS checkpoints
    echo -e "${YELLOW}[1/4] Backing up IndexTTS models...${NC}"
    local indextts_backup="$backup_path/indextts_checkpoints"
    mkdir -p "$indextts_backup"
    
    for file in "${INDEXTTS_FILES[@]}"; do
        local src="$INDEXTTS_CHECKPOINTS/$file"
        if [ -e "$src" ]; then
            cp -r "$src" "$indextts_backup/"
            print_success "$file ($(get_size "$src"))"
        else
            print_warning "$file not found, skipping"
        fi
    done
    
    # Backup LatentSync checkpoints
    echo ""
    echo -e "${YELLOW}[2/4] Backing up LatentSync models...${NC}"
    local latentsync_backup="$backup_path/latentsync"
    mkdir -p "$latentsync_backup/checkpoints/whisper"
    
    for file in "${LATENTSYNC_FILES[@]}"; do
        local src="$LATENTSYNC_DIR/$file"
        if [ -e "$src" ]; then
            cp "$src" "$latentsync_backup/$file"
            print_success "$file ($(get_size "$src"))"
        else
            print_warning "$file not found, skipping"
        fi
    done
    
    # Backup InsightFace models
    echo ""
    echo -e "${YELLOW}[3/4] Backing up InsightFace models...${NC}"
    local insightface_backup="$backup_path/insightface"
    mkdir -p "$insightface_backup"
    
    for model in "${INSIGHTFACE_MODELS[@]}"; do
        local src="$INSIGHTFACE_DIR/$model"
        if [ -d "$src" ]; then
            cp -r "$src" "$insightface_backup/"
            print_success "$model ($(get_size "$src"))"
        else
            print_warning "$model not found, skipping"
        fi
    done
    
    # Backup HuggingFace models (optional)
    echo ""
    echo -e "${YELLOW}[4/4] Backing up HuggingFace cache (optional, may be large)...${NC}"
    read -p "Backup HuggingFace models? This may take a while. (y/N): " confirm
    if [[ "$confirm" =~ ^[Yy]$ ]]; then
        local hf_backup="$backup_path/huggingface"
        mkdir -p "$hf_backup"
        
        for model in "${HF_MODELS[@]}"; do
            local src="$HUGGINGFACE_CACHE/$model"
            if [ -d "$src" ]; then
                cp -r "$src" "$hf_backup/"
                print_success "$model ($(get_size "$src"))"
            else
                print_warning "$model not found, skipping"
            fi
        done
    else
        print_warning "Skipping HuggingFace models"
    fi
    
    # Create metadata
    echo ""
    echo -e "${YELLOW}Creating backup metadata...${NC}"
    cat > "$backup_path/metadata.txt" << EOF
Backup created: $(date)
Host: $(hostname)
User: $(whoami)

IndexTTS Project: $PROJECT_DIR
LatentSync Dir: $LATENTSYNC_DIR
InsightFace Dir: $INSIGHTFACE_DIR

Total backup size: $(get_size "$backup_path")
EOF
    
    print_success "Metadata saved"
    
    # Create symlink to latest backup
    rm -f "$backup_dir/latest"
    ln -sf "$backup_path" "$backup_dir/latest"
    
    echo ""
    print_header "Backup Complete"
    echo "Backup location: $backup_path"
    echo "Total size: $(get_size "$backup_path")"
    echo ""
    echo "To restore, run:"
    echo "  $0 restore $backup_path"
}

restore_models() {
    local backup_path="${1:-$DEFAULT_BACKUP_DIR/latest}"
    
    if [ ! -d "$backup_path" ]; then
        print_error "Backup directory not found: $backup_path"
        exit 1
    fi
    
    print_header "Model Restore"
    echo "Restoring from: $backup_path"
    echo ""
    
    # Show metadata
    if [ -f "$backup_path/metadata.txt" ]; then
        echo -e "${YELLOW}Backup info:${NC}"
        cat "$backup_path/metadata.txt"
        echo ""
    fi
    
    read -p "Continue with restore? (y/N): " confirm
    if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
        echo "Restore cancelled."
        exit 0
    fi
    
    # Restore IndexTTS checkpoints
    echo ""
    echo -e "${YELLOW}[1/4] Restoring IndexTTS models...${NC}"
    local indextts_backup="$backup_path/indextts_checkpoints"
    if [ -d "$indextts_backup" ]; then
        mkdir -p "$INDEXTTS_CHECKPOINTS"
        for file in "$indextts_backup"/*; do
            if [ -e "$file" ]; then
                cp -r "$file" "$INDEXTTS_CHECKPOINTS/"
                print_success "$(basename "$file")"
            fi
        done
    else
        print_warning "IndexTTS backup not found"
    fi
    
    # Restore LatentSync checkpoints
    echo ""
    echo -e "${YELLOW}[2/4] Restoring LatentSync models...${NC}"
    local latentsync_backup="$backup_path/latentsync"
    if [ -d "$latentsync_backup" ]; then
        mkdir -p "$LATENTSYNC_DIR/checkpoints/whisper"
        for file in "${LATENTSYNC_FILES[@]}"; do
            local src="$latentsync_backup/$file"
            if [ -e "$src" ]; then
                cp "$src" "$LATENTSYNC_DIR/$file"
                print_success "$file"
            fi
        done
    else
        print_warning "LatentSync backup not found"
    fi
    
    # Restore InsightFace models
    echo ""
    echo -e "${YELLOW}[3/4] Restoring InsightFace models...${NC}"
    local insightface_backup="$backup_path/insightface"
    if [ -d "$insightface_backup" ]; then
        mkdir -p "$INSIGHTFACE_DIR"
        for model in "$insightface_backup"/*; do
            if [ -d "$model" ]; then
                cp -r "$model" "$INSIGHTFACE_DIR/"
                print_success "$(basename "$model")"
            fi
        done
    else
        print_warning "InsightFace backup not found"
    fi
    
    # Restore HuggingFace models
    echo ""
    echo -e "${YELLOW}[4/4] Restoring HuggingFace models...${NC}"
    local hf_backup="$backup_path/huggingface"
    if [ -d "$hf_backup" ]; then
        mkdir -p "$HUGGINGFACE_CACHE"
        for model in "$hf_backup"/*; do
            if [ -d "$model" ]; then
                cp -r "$model" "$HUGGINGFACE_CACHE/"
                print_success "$(basename "$model")"
            fi
        done
    else
        print_warning "HuggingFace backup not found (this is normal if skipped during backup)"
    fi
    
    echo ""
    print_header "Restore Complete"
}

list_backups() {
    local backup_dir="${1:-$DEFAULT_BACKUP_DIR}"
    
    print_header "Available Backups"
    echo "Backup directory: $backup_dir"
    echo ""
    
    if [ ! -d "$backup_dir" ]; then
        print_warning "Backup directory does not exist"
        exit 0
    fi
    
    local count=0
    for backup in "$backup_dir"/backup_*; do
        if [ -d "$backup" ]; then
            count=$((count + 1))
            local name=$(basename "$backup")
            local size=$(get_size "$backup")
            local date=$(echo "$name" | sed 's/backup_\([0-9]\{8\}\)_\([0-9]\{6\}\)/\1 \2/' | sed 's/\([0-9]\{4\}\)\([0-9]\{2\}\)\([0-9]\{2\}\)/\1-\2-\3/')
            
            if [ -L "$backup_dir/latest" ] && [ "$(readlink "$backup_dir/latest")" = "$backup" ]; then
                echo -e "${GREEN}* $name${NC} - $size (latest)"
            else
                echo "  $name - $size"
            fi
        fi
    done
    
    if [ $count -eq 0 ]; then
        print_warning "No backups found"
    else
        echo ""
        echo "Total: $count backup(s)"
    fi
}

show_usage() {
    echo "Usage: $0 <command> [backup_dir]"
    echo ""
    echo "Commands:"
    echo "  backup   - Create a new backup of all models"
    echo "  restore  - Restore models from a backup"
    echo "  list     - List available backups"
    echo ""
    echo "Options:"
    echo "  backup_dir - Directory to store/read backups (default: ~/model_backups)"
    echo ""
    echo "Examples:"
    echo "  $0 backup                    # Backup to ~/model_backups"
    echo "  $0 backup /mnt/nas/backups   # Backup to custom location"
    echo "  $0 restore                   # Restore from latest backup"
    echo "  $0 restore ~/model_backups/backup_20231221_120000"
    echo "  $0 list                      # List all backups"
}

# Main
case "${1:-}" in
    backup)
        backup_models "$2"
        ;;
    restore)
        restore_models "$2"
        ;;
    list)
        list_backups "$2"
        ;;
    -h|--help|help)
        show_usage
        ;;
    *)
        show_usage
        exit 1
        ;;
esac
