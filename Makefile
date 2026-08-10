# ============================================================
# WM Dynamics RSA - Makefile
# 用法: make [target] [PIPELINE=xxx] [SUBJ=xxx]
# ============================================================

.PHONY: help all install run clean test debug

# 默认目标：显示帮助
help:
	@echo "WM Dynamics RSA - 可用命令:"
	@echo "  make install          创建Conda环境"
	@echo "  make run              运行完整流水线 (默认被试 001)"
	@echo "  make run PIPELINE=rsa SUBJ=002  运行指定阶段"
	@echo "  make debug            快速调试模式 (被试 001, RSA分析)"
	@echo "  make test             运行单元测试"
	@echo "  make clean            清理缓存和临时文件"
	@echo "  make help             显示此帮助信息"

# 创建环境
install:
	conda env create -f environment.yml
	@echo "环境创建完成，请执行: conda activate wm_dynamics"

# 运行完整流水线（支持参数透传）
run:
	python run.py --pipeline $(PIPELINE) --subject $(SUBJ)

# 默认运行完整流水线（全部步骤，被试001）
run-default:
	python run.py --pipeline all --subject 001

# 调试模式：只跑RSA，被试001，详细日志
debug:
	python run.py --pipeline empirical --subject 001 --verbose

# 单元测试
test:
	pytest tests/ -v

# 清理缓存
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name ".DS_Store" -delete
	rm -rf .pytest_cache/ .mypy_cache/ 2>/dev/null || true
	@echo "清理完成"

# 预览（模拟运行，不实际执行）
dry-run:
	python run.py --pipeline all --dry-run

# 更新依赖
update-deps:
	conda env update -f environment.yml --prune

# 下载数据
download:
	python run.py --download

# 下载单个被试（调试用）
download-subj:
	python run.py --download --download-subject $(SUBJ)