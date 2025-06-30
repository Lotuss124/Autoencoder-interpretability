# 加载必要的库
library(ggplot2)
library(dplyr)
library(RColorBrewer)

# 1. 数据预处理 --------------------------------------------------------------
plot_data <- read.csv("neuron_interpretability/plotting_code/3.csv") %>%
  mutate(
    coefficient_of_correlation = as.numeric(coefficient_of_correlation),
    layer = factor(layer, 
                   levels = c("Encoder_0", "Encoder_4", "Encoder_8", 
                              "Decoder_0", "Decoder_4", "Decoder_8")),
    bn = factor(bn, levels = c("no", "yes"), 
                labels = c("Without BN", "With BN"))
  ) %>%
  filter(!is.na(coefficient_of_correlation))

# 2. 绘图 ------------------------------------------------------
p <- ggplot(plot_data, aes(x = bn, y = coefficient_of_correlation, fill = bn)) +
  facet_wrap(~layer, nrow = 1, strip.position = "bottom") +
  
  # 可视化元素
  geom_violin(width = 0.7, alpha = 0.8, color = NA) + 
  geom_boxplot(width = 0.15, fill = "white", 
               outlier.size = 1.5, outlier.shape = 21, outlier.fill = "gray50") +
  
  # 科研级配色方案
  scale_fill_manual(values = c("#4E79A7", "#F28E2B"), guide = "none") + 
  
  # 仅保留必要的轴标签
  labs(
    x = "Batch Normalization",
    y = "Neuron Interpretability Score",
    title = NULL,        # 去除标题
    subtitle = NULL,     # 去除副标题
    caption = NULL       # 去除脚注
  ) +
  
  # 主题设置
  theme_classic(base_size = 11) +
  theme(
    panel.background = element_rect(fill = "white"),
    plot.background = element_rect(fill = "white"), 
    panel.grid.major.y = element_line(color = "gray90", linewidth = 0.25),
    strip.background = element_blank(),
    strip.text = element_text(face = "bold", size = 10, margin = margin(t = 10, b = 10)),
    axis.line = element_line(linewidth = 0.5, color = "black"),
    axis.text = element_text(color = "black"), 
    axis.title = element_text(face = "bold"), 
    panel.spacing = unit(1.2, "lines"),
    plot.title = element_blank(),        # 确保标题空间被移除
    plot.subtitle = element_blank(),    # 确保副标题空间被移除
    plot.caption = element_blank(),     # 确保脚注空间被移除
    axis.text.x = element_text(hjust = 1, vjust = 0.5)
  ) +
  
  scale_y_continuous(breaks = seq(-1, 1, 0.5),
                     limits = c(-1, 1)) +
  coord_cartesian(ylim = c(-1, 1), clip = "off")

# 3. 保存为PDF --------------------------------------------------
ggsave(
  filename = "Figure3a.pdf",
  plot = p,
  device = "pdf",
  width = 10,
  height = 4,
  units = "in",
  dpi = 600,
  bg = "white",
  encoding = "ISOLatin9.enc"
)