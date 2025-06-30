library(ggplot2)
library(Rtsne)
library(readr)
library(patchwork)  # 用于拼图
library(scales)
library(dplyr)

# 设置路径（请替换为实际路径）
deep_feature_path <- "neuron_interpretability/plotting_code/encoder_layer_9_output.csv"
shallow_feature_path <- "neuron_interpretability/plotting_code/encoder_layer_17_output.csv"
label_path <- "neuron_interpretability/input_data/tsne_label.csv"

# 通用数据准备函数
prepare_tsne <- function(feature_path, label_path) {
  # 读取特征数据
  features <- read_csv(
    feature_path,
    col_names = TRUE,
    show_col_types = FALSE
  ) %>%
    rename(tissue_name = 1) %>%
    tibble::column_to_rownames("tissue_name") %>%
    as.matrix()
  
  # 读取标签数据
  labels <- read_csv(
    label_path,
    col_names = FALSE,
    show_col_types = FALSE
  ) %>%
    rename(tissue_name = 1, label = 2) %>%
    tibble::column_to_rownames("tissue_name") %>%
    pull(label)
  
  # 运行t-SNE
  set.seed(42)  # 固定随机种子
  tsne_result <- Rtsne(
    features,
    dims = 2,
    perplexity = 30,
    max_iter = 1000,
    pca = FALSE
  )
  
  # 返回结果数据框
  data.frame(
    tsne1 = tsne_result$Y[,1],
    tsne2 = tsne_result$Y[,2],
    label = factor(labels),
    row.names = rownames(features)
  )
}

# 准备两组数据
deep_data <- prepare_tsne(deep_feature_path, label_path) %>% mutate(group = "Deep")
shallow_data <- prepare_tsne(shallow_feature_path, label_path) %>% mutate(group = "Shallow")

# 创建统一的颜色映射（与Python的husl一致）
n_colors <- length(unique(deep_data$label))
husl_palette <- scales::hue_pal(
  h = c(0, 360) + 15,
  c = 100,
  l = 65,
  h.start = 0,
  direction = 1
)(n_colors)

# 自定义绘图函数
plot_tsne <- function(data, title) {
  ggplot(data, aes(x = tsne1, y = tsne2, color = label)) +
    geom_point(
      size = 3.5,          # 外圈白边
      color = "white",
      alpha = 1
    ) +
    geom_point(
      aes(fill = label),   # 内部彩色
      size = 3,
      shape = 21,
      stroke = 0.3
    ) +
    scale_fill_manual(values = husl_palette) +
    labs(title = title) +
    theme_void() +
    theme(
      plot.title = element_text(hjust = 0.5, size = 12, face = "bold"),
      legend.position = "none",
      plot.margin = margin(5, 5, 5, 5)
    ) +
    coord_fixed()  # 保持纵横比一致
}

# 生成左右两图
p_left <- plot_tsne(deep_data, "Deep Model") 
p_right <- plot_tsne(shallow_data, "Shallow Model")

# 组合图形并添加共享标签
combined_plot <- (p_left + p_right) +
  plot_annotation(
    theme = theme(
      plot.background = element_rect(fill = "white", color = NA)
    )
  ) &
  labs(x = "t-SNE 1", y = "t-SNE 2")  # 共享坐标轴标签

# 保存输出
ggsave(
  "dual_tsne_comparison.pdf",
  combined_plot,
  width = 14,  # 加宽画布
  height = 6,
  device = "pdf"
)

ggsave(
  "dual_tsne_comparison.png",
  combined_plot,
  width = 14,
  height = 6,
  dpi = 300,
  bg = "white"
)