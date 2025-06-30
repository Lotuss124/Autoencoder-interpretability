library(tidyverse)
library(ggpubr)
library(scales)

# 配色方案
model_colors <- c(
  "large" = "#2c6eb3",  # 蓝色
  "small" = "#f0a05a"   # 橙色
)
mean_color <- "#e46247" # 红色
df <- read.csv("neuron_interpretability/plotting_code/5_large.csv")
# 数据处理保持不变
plot_data <- df %>% 
  filter(layer_name == "Decoder_0") %>%
  group_by(model_type, model_id) %>%
  summarise(median_corr = median(correlation_coef, na.rm = TRUE), .groups = "drop") %>%
  mutate(model_type = factor(model_type, levels = c("large", "small")))
# 计算并打印各组统计量
stats_table <- plot_data %>%
  group_by(model_type) %>%
  summarise(
    Median = median(median_corr),
    Q1 = quantile(median_corr, 0.25),
    Q3 = quantile(median_corr, 0.75),
    .groups = "drop"
  )

print("Median and IQR statistics:")
print(stats_table)
# 创建优化后的紧凑图表
p <- ggplot(plot_data, aes(x = model_type, y = median_corr)) +
  # 调整箱线图宽度使其更紧凑
  geom_boxplot(
    aes(color = model_type),
    fill = "white",
    width = 0.4,  # 减小宽度 (原为0.5)
    alpha = 0.8,
    outlier.shape = NA,
    fatten = 2,
    size = 0.6  # 减小边框线粗细
  ) +
  # 调整数据点分布范围
  geom_point(
    aes(color = model_type),
    position = position_jitter(
      width = 0.1,  # 减小抖动范围 (原为0.15)
      height = 0
    ),
    size = 1.8,  # 稍减小点大小
    alpha = 0.7,
    stroke = 0.3
  ) +
  # 均值点保持不变
  stat_summary(
    fun = mean,
    geom = "point",
    shape = 18,
    size = 3,
    color = mean_color,
    stroke = 1.0
  ) +
  # 应用配色
  scale_color_manual(values = model_colors) +
  # 坐标轴和标签
  labs(
    x = "Data Scale",
    y = "Median Neuron Correlation Coefficient"
  ) +
  # 使用更紧凑的主题设置
  theme_pubr() +
  theme(
    legend.position = "none",
    axis.line = element_line(color = "grey30", linewidth = 0.5),
    axis.ticks = element_line(color = "grey30", linewidth = 0.5),
    axis.text = element_text(color = "grey20", size = 10),
    axis.title = element_text(color = "grey10", size = 11, face = "bold"),
    plot.margin = margin(5, 5, 5, 5, unit = "mm"),  # 减小边距
    panel.spacing = unit(0.5, "lines")  # 减小面板间距
  ) +
  # 调整纵轴范围
  scale_y_continuous(
    expand = expansion(mult = c(0.05, 0.05))  # 减小上下扩展空间
  )

# 保存为更紧凑的尺寸
ggsave(
  filename = "pdf/Figure5.pdf",
  plot = p,
  device = "pdf",
  width = 5,  # 减小宽度 (原为8)
  height = 6,
  units = "in",
  dpi = 300
)
