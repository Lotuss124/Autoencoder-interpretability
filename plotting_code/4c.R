library(tidyverse)
library(scales)

# 加载数据（沿用您的预处理代码）
neuron_df <- read_csv("neuron_interpretability/plotting_code/experiment_4.csv") %>%
  mutate(
    activation = factor(activation, levels = c("LeakyReLU", "Tanh", "Sigmoid", "ReLU", "Swish")),
    bn_position = factor(bn_position, 
                         levels = c("after_bn","before_bn"),
                         labels = c("Pre-Activation BN", "Post-Activation BN")),
    corr_level = cut(abs(correlation), 
                     breaks = c(0, 0.3, 0.5, 0.7, 1),
                     labels = c("0-0.3", "0.3-0.5", "0.5-0.7", ">0.7"))
  )

# 计算百分比
stack_data <- neuron_df %>%
  count(activation, bn_position, corr_level) %>%
  group_by(activation, bn_position) %>%
  mutate(percent = n / sum(n) * 100) %>%
  ungroup()

# 绘图
p <- ggplot(stack_data, aes(x = activation, y = percent, fill = corr_level)) +
  # 堆叠柱状图（白色边框）
  geom_col(
    position = position_stack(reverse = TRUE),
    color = "white", linewidth = 0.3, width = 0.7
  ) +
  # 标签（仅显示>5%的数值）
  geom_text(
    aes(label = ifelse(percent > 5, sprintf("%.1f%%", percent), "")),
    position = position_stack(reverse = TRUE, vjust = 0.5),
    size = 3, color = "black"
  ) +
  # 分面
  facet_wrap(~ bn_position, nrow = 1) +
  # 柔和蓝色渐变（四种相近的蓝色）
  scale_fill_manual(
    values = c("#F2F8FD", "#E3EEFB", "#C1DEF6", "#8BC3EE"),
    na.value = "#DADEE0",  # 将NA的灰色改为更浅的灰色（原默认是grey50）
    name = "Correlation Strength",
    labels = c("0-0.3", "0.3-0.5", "0.5-0.7", ">0.7")
  ) +
  # 坐标轴和标签
  labs(x = "Activation Function", y = "Percentage of Neurons") +
  # 主题调整
  theme_minimal(base_size = 12) +
  theme(
    legend.position = "top",
    legend.justification = "center",
    panel.grid.major.x = element_blank(),
    panel.grid.minor.y = element_blank(),
    strip.text = element_text(face = "bold", size = 12, margin = margin(b = 8)),
    axis.text = element_text(color = "black"),
    axis.text.x = element_text(hjust = 0.5, vjust = 1, margin = margin(t = -3, unit = "mm")),
    panel.background = element_rect(fill = "white", color = NA),
    plot.margin = margin(5, 5, 5, 5, unit = "mm")
  )
p
ggsave("Figure4c.pdf", 
       plot = p,
       width = 8, 
       height = 5,
       units = "in",
       dpi = 600,
       bg = "white")