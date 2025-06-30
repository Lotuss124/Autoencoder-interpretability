library(tidyverse)
library(ggpubr)
library(scales)

# 加载数据
df <- read.csv("neuron_interpretability/plotting_code/5_deep.csv")

# 数据处理
plot_data <- df %>%
  filter(!is.na(correlation_coef)) %>%
  group_by(model_depth, model_id) %>%
  summarise(prop_gt_0.5 = mean(correlation_coef > 0.5), .groups = "drop") %>%
  mutate(
    model_num = as.numeric(gsub("model_", "", model_id)),
    model_depth = factor(model_depth, levels = c("deep", "shallow"))
  )
# 计算并打印各组统计量
stats_table <- plot_data %>%
  group_by(model_depth) %>%
  summarise(
    Median = median(prop_gt_0.5),
    Q1 = quantile(prop_gt_0.5, 0.25),
    Q3 = quantile(prop_gt_0.5, 0.75),
    .groups = "drop"
  )

print("Median and IQR statistics:")
print(stats_table)
# 配色方案（保持与参考图一致）
depth_colors <- c(
  "deep" = "#2c6eb3",  # 蓝色
  "shallow" = "#f0a05a"   # 橙色
)
mean_color <- "#e46247" # 红色

# 创建紧凑型箱线图
p <- ggplot(plot_data, aes(x = model_depth, y = prop_gt_0.5)) +
  # 箱线图（参考宽度和样式）
  geom_boxplot(
    aes(color = model_depth),
    fill = "white",
    width = 0.4,        # 紧凑宽度
    alpha = 0.8,
    outlier.shape = NA,
    fatten = 2,         # 加粗中位线
    size = 0.6          # 边框线粗细
  ) +
  # 数据点（调整抖动范围）
  geom_point(
    aes(color = model_depth),
    position = position_jitter(
      width = 0.1,      # 减小抖动范围
      height = 0
    ),
    size = 1.8,         # 点大小
    alpha = 0.7,
    stroke = 0.3
  ) +
  # 均值标记（红色菱形）
  stat_summary(
    fun = mean,
    geom = "point",
    shape = 18,         # 菱形
    size = 3,
    color = mean_color,
    stroke = 1.0
  ) +
  # 应用配色
  scale_color_manual(values = depth_colors) +
  # 坐标轴标签
  labs(
    x = "Model Depth",
    y = "Proportion of Neurons with Correlation > 0.5"
  ) +
  # 紧凑主题设置
  theme_pubr() +
  theme(
    legend.position = "none",
    axis.line = element_line(color = "grey30", linewidth = 0.5),
    axis.ticks = element_line(color = "grey30", linewidth = 0.5),
    axis.text = element_text(color = "grey20", size = 10),
    axis.title = element_text(color = "grey10", size = 11, face = "bold"),
    plot.margin = margin(5, 5, 5, 5, unit = "mm")  # 紧凑边距
  ) +
  # 调整纵轴范围
  scale_y_continuous(
    expand = expansion(mult = c(0.05, 0.05))  # 减小空白区域
  )
p
# 保存为紧凑尺寸
ggsave(
  filename = "Figure5c.pdf",
  plot = p,
  device = "pdf",
  width = 5,  # 紧凑宽度
  height = 6,
  units = "in",
  dpi = 300
)