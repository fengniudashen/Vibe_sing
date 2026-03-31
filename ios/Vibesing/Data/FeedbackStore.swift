import Foundation
import SQLite3

/// Stores user feedback ("Was this accurate?") locally in SQLite.
/// Each record includes: timestamp, user verdict (accurate/inaccurate),
/// and associated DSP features at the moment of detection.
///
/// Data is stored in the app's Documents directory and can be exported
/// for training data collection.
final class FeedbackStore: ObservableObject {
    
    @Published var totalFeedbacks: Int = 0
    @Published var accurateCount: Int = 0
    @Published var inaccurateCount: Int = 0
    
    private var db: OpaquePointer?
    private let dbPath: String
    
    init() {
        let documentsDir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        dbPath = documentsDir.appendingPathComponent("vibesing_feedback.sqlite3").path
        openDatabase()
        createTableIfNeeded()
        refreshCounts()
    }
    
    deinit {
        sqlite3_close(db)
    }
    
    // MARK: - Public
    
    /// Record a user feedback event.
    func recordFeedback(
        accurate: Bool,
        f0: Float = 0,
        hnr: Float = 0,
        energy: Float = 0,
        squeezeProbability: Float = 0
    ) {
        let timestamp = ISO8601DateFormatter().string(from: Date())
        let sql = """
            INSERT INTO feedback (timestamp, accurate, f0, hnr, energy, squeeze_prob)
            VALUES (?, ?, ?, ?, ?, ?)
            """
        
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else { return }
        defer { sqlite3_finalize(stmt) }
        
        sqlite3_bind_text(stmt, 1, (timestamp as NSString).utf8String, -1, nil)
        sqlite3_bind_int(stmt, 2, accurate ? 1 : 0)
        sqlite3_bind_double(stmt, 3, Double(f0))
        sqlite3_bind_double(stmt, 4, Double(hnr))
        sqlite3_bind_double(stmt, 5, Double(energy))
        sqlite3_bind_double(stmt, 6, Double(squeezeProbability))
        
        sqlite3_step(stmt)
        refreshCounts()
    }
    
    /// Export all feedback as CSV string (for batch upload or debugging).
    func exportCSV() -> String {
        var csv = "timestamp,accurate,f0,hnr,energy,squeeze_prob\n"
        let sql = "SELECT timestamp, accurate, f0, hnr, energy, squeeze_prob FROM feedback ORDER BY rowid"
        
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else { return csv }
        defer { sqlite3_finalize(stmt) }
        
        while sqlite3_step(stmt) == SQLITE_ROW {
            let ts = String(cString: sqlite3_column_text(stmt, 0))
            let acc = sqlite3_column_int(stmt, 1)
            let f0 = sqlite3_column_double(stmt, 2)
            let hnr = sqlite3_column_double(stmt, 3)
            let energy = sqlite3_column_double(stmt, 4)
            let prob = sqlite3_column_double(stmt, 5)
            csv += "\(ts),\(acc),\(f0),\(hnr),\(energy),\(prob)\n"
        }
        
        return csv
    }
    
    /// Current accuracy rate based on user feedback.
    var accuracyRate: Double {
        guard totalFeedbacks > 0 else { return 0 }
        return Double(accurateCount) / Double(totalFeedbacks)
    }
    
    // MARK: - Private
    
    private func openDatabase() {
        if sqlite3_open(dbPath, &db) != SQLITE_OK {
            print("FeedbackStore: Failed to open database at \(dbPath)")
        }
    }
    
    private func createTableIfNeeded() {
        let sql = """
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                accurate INTEGER NOT NULL,
                f0 REAL DEFAULT 0,
                hnr REAL DEFAULT 0,
                energy REAL DEFAULT 0,
                squeeze_prob REAL DEFAULT 0
            )
            """
        sqlite3_exec(db, sql, nil, nil, nil)
    }
    
    private func refreshCounts() {
        totalFeedbacks = countRows(where: nil)
        accurateCount = countRows(where: "accurate = 1")
        inaccurateCount = countRows(where: "accurate = 0")
    }
    
    private func countRows(where clause: String?) -> Int {
        let sql = "SELECT COUNT(*) FROM feedback" + (clause.map { " WHERE \($0)" } ?? "")
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else { return 0 }
        defer { sqlite3_finalize(stmt) }
        
        if sqlite3_step(stmt) == SQLITE_ROW {
            return Int(sqlite3_column_int(stmt, 0))
        }
        return 0
    }
}
