library(tidyverse)
library(scales)

neuron_df <- read_csv("neuron_interpretability/plotting_code/4.csv") %>%
  mutate(
    activation = factor(activation, levels = c("LeakyReLU", "Tanh", "Sigmoid", "ReLU", "Swish")),
    bn_position = factor(bn_position, 
                         levels = c("after_bn", "before_bn"),
                         labels = c("Pre-Activation BN", "Post-Activation BN"))
  )
my_colors <- c(
  "LeakyReLU" = "#4e79a7",  # 蓝色
  "Tanh"      = "#59a14f",  # 橙色
  "Sigmoid"   = "#e15759",  # 红色
  "ReLU"      = "#79706e",  # 绿色
  "Swish"     = "#b07aa1"   # 紫色
)

# 修改为箱线图版本（移除了geom_violin，调整了geom_boxplot参数）
p_boxplot <- ggplot(neuron_df, aes(x = activation, y = correlation)) +
  # 彩色边框箱线图（空心填充）
  geom_boxplot(
    aes(color = activation),  # 边框颜色映射
    fill = "white",           # 空心填充
    width = 0.5,
    size = 0.8,               # 边框粗细
    outlier.shape = NA        # 隐藏离群点
  ) +
  # 黑色菱形均值点
  stat_summary(
    fun = mean,               # 计算均值（非中位数！）
    geom = "point",
    shape = 18,               # 菱形符号
    size = 3,                 # 点大小
    color = "black",          # 黑色
    position = position_dodge(width = 0.5)
  ) +
  # 应用自定义颜色
  scale_color_manual(values = my_colors, name = "Activation") +
  # 分面与参考线
  facet_wrap(~ bn_position, nrow = 1) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "grey50") +
  # 主题与标签
  labs(x = NULL, y = "Neuron Interpretability Score") +
  theme_minimal() +
  theme(
    legend.position = "top",
    panel.grid.major.x = element_blank()
  )
p_boxplot
# 保存为高分辨率PNG
ggsave("Figure4a.pdf", 
       plot = p_boxplot,
       width = 8, 
       height = 5.5,
       units = "in",
       dpi = 600,
       bg = "white")