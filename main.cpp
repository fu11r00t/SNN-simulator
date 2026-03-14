#include <QApplication>
#include <QWidget>
#include <QPainter>
#include <QTimer>
#include <QMouseEvent>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <array>

// --- конфигурация (масштабируется через константы) ---
namespace config {
constexpr int dt_ms = 200;       // шаг интегрирования, мс (5 Гц)
constexpr int input_count = 5;   // входы (каналы)
constexpr int hidden_count = 10; // скрытый слой
constexpr int output_count = 2;  // выходы (классы)
constexpr int neuron_count = input_count + hidden_count + output_count;
constexpr double threshold = 1.0;   // порог спайка
constexpr double tau_mem = 20.0;    // постоянная мембраны, мс
constexpr double tau_syn = 5.0;     // постоянная синапса, мс
constexpr double reset_val = 0.0;   // сброс потенциала
constexpr int refractory_ms = 30;   // рефрактерный период, мс
constexpr double noise_std = 0.02;  // шум
constexpr double input_gain = 10.0; // усиление внешнего входа
constexpr int max_delay_frames = 3; // макс. задержка синапса, кадров
} // namespace config

struct neuron
{
    double v = 0.0;               // мембранный потенциал
    double i_syn = 0.0;           // синаптический ток
    int refractory = 0;           // остаток рефрактерного периода, мс
    bool spiked = false;          // спайк в текущем кадре
    bool spiked_prev = false;     // спайк в предыдущем кадре
    int layer = 0;                // 0=входной, 1=скрытый, 2=выходной
    int id_in_layer = 0;          // индекс внутри слоя
    double activation_rate = 0.0; // скользящее среднее активности
};

struct synapse
{
    int from, to;
    double weight;
    int delay_frames = 1; // задержка в кадрах (>=1)
};

class snn_widget : public QWidget
{
    Q_OBJECT
public:
    explicit snn_widget(QWidget* parent = nullptr)
        : QWidget(parent)
    {
        setMinimumSize(900, 600);
        setWindowTitle("SNN Demo");
        setAutoFillBackground(true);

        rng.seed(std::random_device{}());
        noise_dist = std::normal_distribution<>(0.0, config::noise_std);

        init_network();

        timer = new QTimer(this);
        connect(timer, &QTimer::timeout, this, &snn_widget::simulate_step);
        timer->start(config::dt_ms);

        setMouseTracking(true);
    }

    // интерфейс для подачи ээг-сигнала (вызывать извне)
    void feed_eeg_input(const std::array<double, config::input_count>& values)
    {
        for (int i = 0; i < config::input_count; ++i) {
            // преобразуем амплитуду ээг в синаптический ток
            neurons[i].i_syn += values[i] * config::input_gain;
        }
    }

    // чтение выхода для классификации: какой выходной нейрон активнее
    int get_prediction() const
    {
        int best = config::input_count + config::hidden_count;
        double best_rate = -1.0;
        for (int i = 0; i < config::output_count; ++i) {
            int idx = config::input_count + config::hidden_count + i;
            if (neurons[idx].activation_rate > best_rate) {
                best_rate = neurons[idx].activation_rate;
                best = idx;
            }
        }
        return best - (config::input_count + config::hidden_count); // 0 или 1
    }

private:
    std::vector<neuron> neurons;
    std::vector<synapse> synapses;
    // очередь спайков: [нейрон][индекс задержки]
    std::vector<std::array<double, config::max_delay_frames>> spike_queue;
    QTimer* timer = nullptr;
    std::mt19937 rng;
    std::normal_distribution<double> noise_dist;
    int hover_neuron = -1;
    int frame_count = 0;

    void init_network()
    {
        neurons.resize(config::neuron_count);
        spike_queue.resize(config::neuron_count);

        // инициализация слоёв
        int idx = 0;
        for (int i = 0; i < config::input_count; ++i)
            neurons[idx].layer = 0, neurons[idx].id_in_layer = i, ++idx;
        for (int i = 0; i < config::hidden_count; ++i)
            neurons[idx].layer = 1, neurons[idx].id_in_layer = i, ++idx;
        for (int i = 0; i < config::output_count; ++i)
            neurons[idx].layer = 2, neurons[idx].id_in_layer = i, ++idx;

        // вход -> скрытый (структурированная связность)
        for (int i = 0; i < config::input_count; ++i) {
            for (int j = 0; j < config::hidden_count; ++j) {
                double w = 0.0;
                if (i < 3 && j < 5)
                    w = 0.5 + 0.1 * ((i + j) % 3);
                else if (i >= 3 && j >= 5)
                    w = 0.5 + 0.1 * ((i + j) % 3);
                else
                    w = 0.05 + 0.03 * (rng() % 10);
                synapses.push_back({i, config::input_count + j, w, 1});
            }
        }
        // скрытый -> выход
        for (int i = 0; i < config::hidden_count; ++i) {
            for (int j = 0; j < config::output_count; ++j) {
                double w = 0.0;
                if (i < 5 && j == 0)
                    w = 0.6 + 0.05 * (i % 4);
                else if (i >= 5 && j == 1)
                    w = 0.6 + 0.05 * (i % 4);
                else
                    w = 0.05 + 0.03 * (rng() % 10);
                synapses.push_back(
                    {config::input_count + i, config::input_count + config::hidden_count + j, w, 1});
            }
        }
    }

    void simulate_step()
    {
        ++frame_count;
        const double dt = config::dt_ms / 1000.0; // секунды

        // 1. сохраняем спайки для задержки
        for (auto& n : neurons) {
            n.spiked_prev = n.spiked;
            n.spiked = false;
        }

        // 2. применяем задержанные спайки к току
        for (int i = 0; i < config::neuron_count; ++i) {
            neurons[i].i_syn += spike_queue[i][0];
            // сдвиг очереди влево
            for (int d = 0; d < config::max_delay_frames - 1; ++d)
                spike_queue[i][d] = spike_queue[i][d + 1];
            spike_queue[i][config::max_delay_frames - 1] = 0.0;
        }

        // 3. генерируем новые спайки в очередь (с задержкой)
        for (const auto& s : synapses) {
            if (neurons[s.from].spiked_prev) {
                int delay = std::min(s.delay_frames, config::max_delay_frames - 1);
                spike_queue[s.to][delay] += s.weight;
            }
        }

        // 4. обновляем нейроны (LIF модель, численно устойчивая)
        for (int i = 0; i < config::neuron_count; ++i) {
            neuron& n = neurons[i];

            if (n.refractory > 0) {
                n.refractory -= config::dt_ms;
                n.v = config::reset_val;
                continue;
            }

            // dv/dt = (-v + i_syn) / tau_mem  =>  dv = dt * (-v + i_syn) / tau_mem
            // все величины в мс, dt переведён в секунды, но отношение сохраняется
            const double dv = dt * (-n.v + n.i_syn) / (config::tau_mem / 1000.0);
            n.v += dv + noise_dist(rng);

            // экспоненциальное затухание синаптического тока
            n.i_syn *= std::exp(-dt / (config::tau_syn / 1000.0));

            if (n.v >= config::threshold) {
                n.spiked = true;
                n.v = config::reset_val;
                n.refractory = config::refractory_ms;
                n.activation_rate = 0.95 * n.activation_rate + 0.05 * 1.0;
            } else {
                n.activation_rate *= 0.99;
            }
        }

        update();
    }

    QPointF get_neuron_pos(int idx, int w, int h) const
    {
        const int layer = neurons[idx].layer;
        const int id = neurons[idx].id_in_layer;
        const int layers = 3;
        const int margin_x = w / 8;
        const int usable_w = w - 2 * margin_x;
        const int x = margin_x + (usable_w * layer) / (layers - 1);

        const int count = (layer == 0)   ? config::input_count
                          : (layer == 1) ? config::hidden_count
                                         : config::output_count;
        const int margin_y = h / 8;
        const int usable_h = h - 2 * margin_y;
        const int y = margin_y + (usable_h * (id + 1)) / (count + 1);

        return {static_cast<double>(x), static_cast<double>(y)};
    }

    int get_neuron_at_pos(const QPoint& pos) const
    {
        const int w = width(), h = height();
        for (int i = 0; i < config::neuron_count; ++i) {
            const auto p = get_neuron_pos(i, w, h);
            const double r = (neurons[i].layer == 0) ? 22.0 : 28.0;
            const double dx = pos.x() - p.x(), dy = pos.y() - p.y();
            if (dx * dx + dy * dy < r * r)
                return i;
        }
        return -1;
    }

    void paintEvent(QPaintEvent*) override
    {
        QPainter p(this);
        p.setRenderHint(QPainter::Antialiasing);
        p.fillRect(rect(), QColor(15, 15, 25));

        const int w = width(), h = height();

        // заголовок
        p.setPen(QColor(200, 200, 255));
        QFont title_font = p.font();
        title_font.setPointSize(14);
        title_font.setBold(true);
        p.setFont(title_font);
        p.drawText(rect(), Qt::AlignTop | Qt::AlignHCenter, "SNN Simulator v2.0");

        // статистика
        p.setFont(QFont("Arial", 9));
        p.setPen(QColor(150, 150, 200));
        const int spikes_now = std::count_if(neurons.begin(), neurons.end(), [](const neuron& n) {
            return n.spiked;
        });
        p.drawText(20,
                   h - 30,
                   QString("frame: %1 | spikes: %2 | fps: ~%3")
                       .arg(frame_count)
                       .arg(spikes_now)
                       .arg(1000 / config::dt_ms));

        // синапсы
        for (const auto& s : synapses) {
            const auto from = get_neuron_pos(s.from, w, h);
            const auto to = get_neuron_pos(s.to, w, h);
            double line_w = 1.0 + s.weight * 4.0;
            QColor col;
            if (s.weight > 0.4)
                col = QColor(80, 220, 80);
            else if (s.weight > 0.2)
                col = QColor(100, 100, 150);
            else
                col = QColor(60, 60, 80);
            if (neurons[s.from].spiked_prev) {
                col = QColor(255, 200, 50);
                line_w += 1.5;
            }
            col.setAlpha(80 + static_cast<int>(s.weight * 120));
            p.setPen(QPen(col, line_w));
            p.drawLine(from, to);
        }

        // нейроны
        for (int i = 0; i < config::neuron_count; ++i) {
            const neuron& n = neurons[i];
            const auto pos = get_neuron_pos(i, w, h);
            const double r = (n.layer == 0) ? 22.0 : 28.0;

            QColor col;
            if (n.refractory > 0)
                col = QColor(50, 50, 70);
            else if (n.spiked)
                col = QColor(255, 255, 200);
            else {
                const int blue = 80 + std::min(175, static_cast<int>(n.v * 100));
                const int green = 50 + std::min(100, static_cast<int>(n.i_syn * 200));
                col = QColor(green, 50, blue);
            }

            p.setPen(i == hover_neuron ? QPen(QColor(255, 255, 100), 4)
                                       : QPen(QColor(150, 150, 200), 2));
            p.setBrush(col);
            p.drawEllipse(pos, r, r);

            // подпись
            p.setPen(Qt::white);
            QFont f = p.font();
            f.setPointSize(9);
            p.setFont(f);
            QString label;
            if (n.layer == 0)
                label = "I" + QString::number(n.id_in_layer);
            else if (n.layer == 2)
                label = "O" + QString::number(n.id_in_layer);
            p.drawText(QRectF(pos.x() - 12, pos.y() - 8, 24, 16), Qt::AlignCenter, label);

            if (n.activation_rate > 0.1) {
                p.setPen(QColor(255, 200, 50));
                p.drawText(pos.x() + 20, pos.y() + 5, QString::number(n.activation_rate, 'f', 2));
            }
        }

        // легенда
        p.setPen(QColor(180, 180, 220));
        p.setFont(QFont("Arial", 9));
        p.drawText(20, 50, "ЛКМ: стимулировать вход | ПКМ: сброс | выходы: классы ээг");
        p.drawText(20, 70, "зелёные связи = сильные | жёлтые = активные");
    }

    void mousePressEvent(QMouseEvent* event) override
    {
        const int idx = get_neuron_at_pos(event->pos());
        if (idx >= 0 && neurons[idx].layer == 0) {
            if (event->button() == Qt::LeftButton) {
                neurons[idx].v += config::threshold * config::input_gain;
                neurons[idx].i_syn += 20.0;
            } else if (event->button() == Qt::RightButton) {
                neurons[idx].v = config::reset_val;
                neurons[idx].i_syn = 0;
            }
            update();
        }
    }

    void mouseMoveEvent(QMouseEvent* event) override
    {
        hover_neuron = get_neuron_at_pos(event->pos());
        update();
    }

    void leaveEvent(QEvent*) override
    {
        hover_neuron = -1;
        update();
    }
};

int main(int argc, char* argv[])
{
    QApplication app(argc, argv);
    app.setStyle("Fusion");
    snn_widget widget;
    widget.show();
    return app.exec();
}

#include "main.moc"
